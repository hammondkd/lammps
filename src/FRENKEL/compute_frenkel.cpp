/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:  Karl D. Hammond <hammondkd@missouri.edu>
                         University of Missouri, Columbia (USA), 2018
------------------------------------------------------------------------- */

#include <mpi.h>
#include "compute_frenkel.h"
#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "create_atoms.h"
#include "delete_atoms.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "lattice.h"
#include "memory.h"
#include "update.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#define Frenkel_latticesites "Frenkel_latticesites"
#define Frenkel_everything "Frenkel_everything"
#define BIN_GROW_SIZE 32 // Lattice bins start this size and grow in this incr.
#define MAX_OCCUPANTS 8 // Maximum number of occupants of one lattice site
#define BIG 1.0E20
#define SMALL 1.0E-10

// C++ does not have an equivalent to Fortran's nint or Python's round, so
// we make one up!
#define nint(x) (static_cast<int>( (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5) ))
#define anint(x) ( (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5) )
#define str(a) xstr(a)
#define xstr(x) #x

// The LAMMPS MPI stubs don't yet include these
#ifdef MPI_STUBS
 #ifndef MPI_PROC_NULL
  #define MPI_PROC_NULL -2
 #endif
 #ifndef MPI_REQUEST_NULL
  #define MPI_REQUEST_NULL ((MPI_Request) 0)
 #endif
#endif

using namespace LAMMPS_NS;
using namespace std;

static const char cite_compute_frenkel_c[] =
  "compute_frenkel command: doi:10.1016/j.cpc.2019.106862\n\n"
  "@article{Hammond2020a,\n"
  "  author  = \"Hammond, Karl D.\",\n"
  "  title   = \"Parallel Point Defect Identification in Molecular Dynamics\n"
  "             Simulations Without Post-Processing: A Compute and Dump Style\n"
  "             for {LAMMPS}\",\n"
  "  journal = \"Computer Physics Communications\",\n"
  "  volume  = 247,\n"
  "  pages   = 106862,\n"
  "  doi     = 10.1016/j.cpc.2019.106862,\n"
  "  year    = 2020,\n"
  "}\n\n";

/* To the unaware:  I found during the course of writing this class that
   there is a difference between abs(x) and std::abs(x) when the cmath header
   is included.  In g++, anyway, abs(x) refers to the C function, which takes
   an int argument and returns an int; std::abs(x), on the other hand, refers
   to the overloaded C++ function, which can take several kinds of arguments
   and return the corresponding data type.  The C++ standard evidently tried
   to denounce this sort of behavior, but was later revised to allow it.  It
   was a very fun afternoon wondering why abs(-18.0) was returning -0.5, or
   perhaps 3.184E-320.  Adding "using namespace std" will fix this problem,
   as will explicitly calling for std::abs or using fabs or std::fabs in its
   place. */

// ComputeFrenkel::ComputeFrenkel () // {{{1
ComputeFrenkel::ComputeFrenkel (class LAMMPS* lmp, int narg, char** arg) :
    Compute (lmp, narg, arg) {

  if ( narg != 3 ) error->all (FLERR, "Illegal compute frenkel command");

  if ( lmp->citeme ) lmp->citeme->add(cite_compute_frenkel_c);

  // Error if no atom tags are defined or there is no atom map
  if ( not atom->tag_enable )
    error->all (FLERR, "Cannot use compute style frenkel unless atoms have IDs");
  // August 25, 2020: added "or domain->lattice-nbasis == 0" to this check
  if ( not domain->lattice or domain->lattice->nbasis == 0 )
    error->all (FLERR,"Use of compute style frenkel with undefined lattice.");

  comm_reverse = 1;
  vector_flag = array_flag = peratom_flag = local_flag = 1;
  size_peratom_cols = 0;
  size_local_rows = 0; // Will eventually change, of course
  size_local_cols = 5; // Cluster tag, cluster size, x, y, z
  size_vector = 3;  // Vacancies (0), interstitials (1), and irregulars (2)
  size_array_rows = 2; // Vacancies and Interstitials
  size_array_cols = 20; // number of columns of output for global array:
                        // clusters with more than this many vacancies or
                        // interstitials will all be counted in the last column
  this->vector_atom = nullptr;
  this->array_local = nullptr;
  extvector = 0;
  extarray = 0;
  memory->create (vector, size_vector, "ComputeFrenkel:vector");
  memory->create (array, size_array_rows, size_array_cols,
    "ComputeFrenkel:array");

  //iregion = -1;
  //idregion = nullptr;
  region = nullptr;
  ifgroup = igroup;
  fgroupbit = groupbit;
  rescale = false;
  sitefile = nullptr;

  nnormal = 0;
  normal = nullptr;

  // Set defaults
  double a_max = MAX( MAX( domain->lattice->xlattice,
      domain->lattice->ylattice), domain->lattice->zlattice);
  cut_vac = 1.01 * a_max;
  cut_int = 1.42 * a_max;
  cutoff = MAX (cut_vac, cut_int);
  binwidth = cutoff;

  nlatsites = 0;
  nlatghosts = 0;
  latsites = latsites0 = nullptr;
  site_tag = nullptr;
  site_mindist = nullptr;
  mindist = nullptr;
  first_local_tag = 0;
  for ( int i = 0; i < 4; i++ )
    nlatbins[i] = 0;
  latbins = nullptr;
  noccupants = nullptr;
  occupant_tag = nullptr;
  noccupied = 0;
  nlist = nullptr;
  clusterID = nullptr;
  cluster_size = nullptr;
  cluster_nsites = nullptr;
  cluster_center = nullptr;
  occupied_cluster_ID = nullptr;

  invoked_find_defects = -1;
  invoked_find_clusters = -1;
  invoked_construct_WS_cell = -1;

}

/****************************************************************************/

ComputeFrenkel::~ComputeFrenkel () { // {{{1

  // I AM DESTRUCTOR!
  // (Sorry to the non-Futurama fans out there who don't get this joke)

  //delete [] idregion;
  delete [] sitefile;

  memory->destroy (vector);
  memory->destroy (array);
  memory->destroy (array_local);

  memory->destroy (latsites);
  memory->destroy (latsites0);
  memory->destroy (site_tag);
  memory->destroy (normal);
  memory->destroy (noccupants);
  memory->destroy (occupant_tag);
  memory->destroy (latbins);
  memory->destroy (mindist);
  memory->destroy (site_mindist);
  memory->destroy (clusterID);
  delete [] cluster_size;
  delete [] cluster_nsites;
  memory->destroy (cluster_center);
  memory->destroy (occupied_cluster_ID);
  delete [] nlist;

}

/****************************************************************************/

void ComputeFrenkel::modify_params (int narg, char** arg) { // {{{1

  if ( narg == 0 ) error->all (FLERR, "Illegal compute_modify command");

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"extra") == 0 ||
        strcmp(arg[iarg],"extra/dof") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute_modify command");
      else error->warning (FLERR,
        "Compute style frenkel does not have degrees of freedom");
      // 01/21/2022: Updated for modified code in compute.cpp
      //extra_dof = force->numeric(FLERR,arg[iarg+1]);
      extra_dof = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"dynamic") == 0 ||
             strcmp(arg[iarg],"dynamic/dof") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute_modify command");
      else error->warning (FLERR,
        "Compute style frenkel does not use the dynamic key word.");
      if (strcmp(arg[iarg+1],"no") == 0) dynamic_user = 0;
      else if (strcmp(arg[iarg+1],"yes") == 0) dynamic_user = 1;
      else error->all(FLERR,"Illegal compute_modify command");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"drvac") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute_modify command");
      cut_vac = atof(arg[iarg+1]);
      if ( cut_vac <= 0.0 ) error->all (FLERR,"Illegal compute_modify command: "
        " cutoff values must be positive.");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"drint") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute_modify command");
      cut_int = atof(arg[iarg+1]);
      if ( cut_vac <= 0.0 ) error->all (FLERR,"Illegal compute_modify command: "
        " dr values must be positive.");
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute_modify command");
      if ( strcmp(arg[iarg+1], "none") == 0 ) {
        region = nullptr;
        //iregion = -1;
        //delete [] idregion;
        //idregion = nullptr;
      }
      else {
        region = domain->get_region_by_id(arg[iarg+1]);
        //iregion = domain->find_region(arg[iarg+1]);
        //if ( iregion == -1 )
        if ( not region )
          error->all (FLERR, "Compute_modify region does not exist");
        //idregion = new char[strlen(arg[iarg+1])+1];
        //strcpy(idregion, arg[iarg+1]);
      }
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"frenkelgroup") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute_modify command");
      ifgroup = group->find(arg[iarg+1]);
      if ( ifgroup == -1 )
        error->all (FLERR,"Compute_modify group does not exist");
      fgroupbit = group->bitmask[ifgroup];
      iarg += 2;
    }
    else if (strcmp(arg[iarg],"rescale") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute_modify command");
      if (strcmp(arg[iarg+1],"no") == 0) rescale = false;
      else if (strcmp(arg[iarg+1],"yes") == 0) rescale = true;
      else error->all(FLERR,"Illegal compute_modify command");
      iarg += 2;
    }
    else if ( strcmp(arg[iarg],"site_file") == 0 ) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute_modify command");
      delete [] sitefile;
      sitefile = nullptr;
      if ( strcmp(arg[iarg+1],"none") != 0 ) {
        sitefile = new char[strlen(arg[iarg+1])+1];
        strcpy (sitefile, arg[iarg+1]);
        FILE* tmp = fopen (sitefile, "r");
        if ( tmp == nullptr ) {
          char errormessage[128];
          sprintf (errormessage,"File %s cannot be opened for reading",
            arg[iarg+1]);
          error->one (FLERR, errormessage);
        }
        else
          fclose (tmp);
      }
      iarg += 2;
    }
    else error->all(FLERR,"Illegal compute_modify command");
  }

}

/****************************************************************************/

void ComputeFrenkel::init () { // {{{1

  // Note that invoked_vector and invoked_array are reset inside
  // modify->compute[]->init during the initialization routines.
  invoked_find_defects = -1;
  invoked_find_clusters = -1;
  invoked_construct_WS_cell = -1;

  // Make sure we have a way to generate lattice sites
  if ( not domain->lattice )
    error->all (FLERR,"Use of compute style frenkel with undefined lattice.");

  create_lattice_sites ();
  put_sites_in_bins ();

  // Initialize occupancy lists for all sites
  memory->destroy (noccupants);
  memory->destroy (occupant_tag);
  memory->create (noccupants, nlatsites, "ComputeFrenkel:noccupants");
  memory->create (occupant_tag, nlatsites, MAX_OCCUPANTS,
    "ComputeFrenkel:occupant_tag");
  for ( int i = 0; i < nlatsites; i++ )
    noccupants[i] = 0;
  // Assumes occupant_tag is contiguous (should be true)
  for ( int i = 0; i < nlatsites * MAX_OCCUPANTS; i++ )
    occupant_tag[0][i] = -1;

  // Check cutoffs, in case the user changed them

  construct_WS_cell ();

  // Build neighbor list, which should never change until the sites do
  exchange_lattice_ghosts (); // should skip occupancies until second one
  construct_site_nlists ();
// FIXME:  Do we really need to do these two things (since we do them again
// when find_defects is called)?

  // Store current box boundaries
  old_boxlo[0] = domain->boxlo[0];
  old_boxlo[1] = domain->boxlo[1];
  old_boxlo[2] = domain->boxlo[2];
  old_boxhi[0] = domain->boxhi[0];
  old_boxhi[1] = domain->boxhi[1];
  old_boxhi[2] = domain->boxhi[2];

}

/****************************************************************************/

int ComputeFrenkel::pack_reverse_comm (int n, int first, double* buf) { // {{{1

  int m, last;
  m = 0;
  last = first + n;
  for ( int i = first; i < last; i++ )
    buf[m++] = mindist[i];
  return m;

}

/****************************************************************************/

// int ComputeFrenkel::unpack_reverse_comm {{{1
void ComputeFrenkel::unpack_reverse_comm (int n, int* list, double* buf) {

  int j = 0, m = 0;
  double tmp;

  for (int i = 0; i < n; i++ ) {
    j = list[i];
    tmp = buf[m++];  // So that m doesn't get incremented TWICE
    mindist[j] = MIN (mindist[j], tmp);
  }

}

/****************************************************************************/

void ComputeFrenkel::find_defects () { // {{{1

  invoked_find_defects = update->ntimestep;

  rescale_lattice_sites ();
  exchange_lattice_ghosts (); // In case sites are NOW close enough to exchange
  construct_site_nlists ();

  // Allocate and zero out mindist, site_mindist, and noccupants
  memory->destroy (site_mindist);
  memory->create (site_mindist, nlatsites, "ComputeFrenkel:site_mindist");
  for ( int k = 0; k < nlatsites; k++ )
    noccupants[k] = 0;
  for ( int n = 0; n < nlatsites * MAX_OCCUPANTS; n++ )
    occupant_tag[0][n] = -1; // assumes occupant_tag is contiguous
  for ( int k = 0; k < nlatsites; k++ )
    site_mindist[k] = BIG;

  memory->destroy (mindist);
  memory->create (mindist, atom->nmax, "ComputeFrenkel:mindist");
  for ( int n = 0; n < atom->nmax; n++ )
    mindist[n] = BIG;

  double cutsq = cutoff * cutoff;

  // Loop over all atoms and ghosts
  for ( int n = 0; n < atom->nlocal + atom->nghost; n++ ) {

    // Find closest lattice bin
    int ii, jj, kk;
    find_closest_bin (atom->x[n], ii, jj, kk);

    // Find closest site within this bin
    double drsq_min = BIG;
    int closest_site = -1;
    double dx, dy, dz, drsq;
    for ( int m = 0; m < nlatbins[3]; m++ ) {
      int s = latbins[ii][jj][kk][m];
      if ( s < 0 ) break; // Reached end of list
      double* r_s = latsites[s];
      dx = r_s[0] - atom->x[n][0];
      dy = r_s[1] - atom->x[n][1];
      dz = r_s[2] - atom->x[n][2];
      // Apply periodic boundary conditions, if necessary
      domain->minimum_image (dx, dy, dz);
      drsq = dx*dx + dy*dy + dz*dz;
      if ( drsq < drsq_min and drsq <= cutsq) {
        drsq_min = drsq;
        closest_site = s;
      }
    }

    // Now loop over the neighbor list of that site, to find ghosts that
    // might be even closer (ghosts are not always in the right bin)
    if ( closest_site >= 0 ) {
      int closest_site_before = closest_site;
      do {
        closest_site_before = closest_site;
        std::list<tagint>::iterator item, end;
        end = nlist[closest_site].end();
        for ( item = nlist[closest_site].begin() ; item != end ; item++ ) {
          int s = site_tag2index (*item);
          if ( s < 0 ) continue; // Reached end of list
          double* r_s = latsites[s];
          dx = r_s[0] - atom->x[n][0];
          dy = r_s[1] - atom->x[n][1];
          dz = r_s[2] - atom->x[n][2];
          // Apply periodic boundary conditions, if necessary
          domain->minimum_image (dx, dy, dz);
          drsq = dx*dx + dy*dy + dz*dz;
          if ( drsq < drsq_min and drsq <= cutsq) {
            drsq_min = drsq;
            closest_site = s;
            break;
          }
        }
      } while ( closest_site != closest_site_before );
    }

    // If a nearby site was found on this process, set mindist
    // Otherwise, leave it set to BIG (implies it's on another process)
    if ( closest_site < 0 ) {
      mindist[n] = BIG;
      continue;
    }
    else
      mindist[n] = sqrt(drsq_min);

    site_mindist[closest_site] = MIN (site_mindist[closest_site], mindist[n]);

    // If inside the Wigner-Seitz cell, add it to the occupancy list (if it's
    // not already in it)
    if ( inside_WS_cell (n, closest_site) ) {
      // If already in site's occupancy list, we're done
      if ( tag_is_already_in_occupancy_list (atom->tag[n], closest_site) )
        continue;
      // If not, add it to the list and increase site occupancy
      noccupants[closest_site] += 1;
      int s = next_free_occupant_tag_index (closest_site, __LINE__);
      occupant_tag[closest_site][s] = atom->tag[n];
    }
  }

  // Exchange mindist info amongst processes for ghost atoms
  //comm->reverse_comm_compute (this);
  comm->reverse_comm (this);

}

/****************************************************************************/

void ComputeFrenkel::find_clusters () { // {{{1

  invoked_find_clusters = update->ntimestep;

  memory->destroy (clusterID);
  memory->create (clusterID, nlatsites, "ComputeFrenkel:clusterID");

  // Exchange "ghost" sites with nearby processes
  // includes site tag, cluster ID, position, and number of occupants
//  exchange_lattice_ghosts ();

  // Start each defect off in its own cluster with cluster ID = site ID
  for ( int k = 0; k < nlatsites; k++ ) {
    if ( noccupants[k] == 1 )
      clusterID[k] = 0;  // "regular" sites
    else
      clusterID[k] = site_tag[k]; // interstitials, vacancies, and "irregulars"
  }

  std::list<tagint>::iterator item, end;

  double dx, dy, dz, drsq;
  double cutvacsq = cut_vac * cut_vac;
  double cutintsq = cut_int * cut_int;

  // For each local site, parse the neighbor list to find clusters.  Repeat
  // until there are no changes on any process.
  int changes_made, global_changes_made, done;
  global_changes_made = false;
  do {
    exchange_lattice_ghosts ();
    changes_made = false;
    do {
      done = true;
      for ( int n = 0; n < nlatsites + nlatghosts ; n++ ) {
        if ( noccupants[n] == 1 ) continue; // Skip "normal" sites
        if ( nlist[n].empty() ) continue;
        item = nlist[n].begin();
        end = nlist[n].end();
        while ( item != end ) {
          int m = site_tag2index (*item);
          if ( m != n and m >= 0 and
              noccupants[m] != 1 and clusterID[m] != clusterID[n] ) {
            dx = latsites[n][0] - latsites[m][0];
            dy = latsites[n][1] - latsites[m][1];
            dz = latsites[n][2] - latsites[m][2];
            domain->minimum_image (dx, dy, dz);
            drsq = dx*dx + dy*dy + dz*dz;
            if ( (noccupants[m] >= 2 and drsq <= cutintsq) or
                 (noccupants[m] == 0 and drsq <= cutvacsq) ) {
              clusterID[n] = clusterID[m] = MIN(clusterID[n], clusterID[m]);
              done = false;
            }
          }
          item++;
        }
      }
      if ( not done ) changes_made = true;
    } while ( not done );

    // We stop only if NO process has changes in its cluster assignments
    MPI_Allreduce (&changes_made, &global_changes_made, 1, MPI_INT,
      MPI_LOR, world);

  } while ( global_changes_made );

  // Find the size of each cluster by counting the sites in each cluster;
  // vacancies are counted as -1 and interstitials as +1.

  // This algorithm is O(nlatsites_global), or O(natoms), in memory if we do
  // it with global site IDs.  This works great for small systems, but if
  // you have a billion atoms, it sort of defeats the purpose of
  // parallelization.... Instead, we generate an array containing a list of
  // all cluster IDs that are non-zero and store ONLY those cluster sizes and
  // centers of mass.
  find_occupied_clusters ();

  // If we found NO occupied clusters, we're done
  if ( noccupied == 0 ) return;

  // Find the number of sites involved in each cluster (its size), and the
  // center of each cluster (NOT the center of mass, the center)
  int* local_cluster_size = new int[noccupied] ();
  int* local_cluster_nsites = new int[noccupied] ();
  double** local_cluster_xi;
  double** local_cluster_zeta;
  memory->create (local_cluster_xi, noccupied, 3,
    "ComputeFrenkel::local_cluster_xi");
  memory->create (local_cluster_zeta, noccupied, 3,
    "ComputeFrenkel::local_cluster_zeta");
  for ( int i = 0; i < noccupied * 3; i++ )
    local_cluster_xi[0][i] = local_cluster_zeta[0][i] = 0.0;
  int n = 0;
  double theta;
  for ( int k = 0; k < nlatsites; k++ ) {
    if ( clusterID[k] == 0 ) continue;
    n = clusterID2occupied_index (clusterID[k]);
    if ( n < 0 ) {
      // should NEVER happen...
      error->warning (FLERR, "Did not find cluster index");
      continue;
    }
    if ( noccupants[k] == 0 ) // vacancy
      local_cluster_size[n] += -1;
    else if ( noccupants[k] >= 2 ) // interstitial
      local_cluster_size[n] += +1;

    // This calculates the center of mass using the image as determined from
    // the method of Bai and Breen (J. Graph. Tools 13(4): 53-60 (2008).
    // Note that their method does NOT calculate the center of mass;
    // here's what I actually do:
    //  (1) For each spatial dimension, wrap the simulation box around the
    //      complex unit circle:  z_j = exp(2*pi*i*x_j/L) = xi_j + i*zeta_j
    //  (2) What we want is the /geometric/ mean,
    //      <z> = prod_{j=1}^N z_j^(1/N)
    //  but that has N-1 non-degenerate branches, all of which produce a
    //  different value of <z>!  So we need to pick the "right" geometric mean
    //  (3) Calculate the /arithmetic/ mean in the complex plane,
    //      <z>_a = sum_{j=1}^N z_j/N
    //  (4) Back out <x>_a, the arithmetic mean position, via
    //      <x>_a = arg(<z>_a)
    //  (5) Calculate the "correct" center of mass by taking the image closest
    //    to <x>_a for each particle and averaging in the usual way
    if ( noccupants[k] != 1 ) {
      local_cluster_nsites[n] += 1;
      if ( domain->xperiodic ) {
        theta = (latsites[k][0] - domain->boxlo[0]) / domain->xprd * 2*M_PI;
        local_cluster_zeta[n][0] += sin(theta);
        local_cluster_xi[n][0] += cos(theta);
      }
      else // If not periodic, xi = x (no projection required)
        local_cluster_xi[n][0] += latsites[k][0];
      if ( domain->yperiodic ) {
        theta = (latsites[k][1] - domain->boxlo[1]) / domain->yprd * 2*M_PI;
        local_cluster_zeta[n][1] += sin(theta);
        local_cluster_xi[n][1] += cos(theta);
      }
      else
        local_cluster_xi[n][1] += latsites[k][1];
      if ( domain->zperiodic ) {
        theta = (latsites[k][2] - domain->boxlo[2]) / domain->zprd * 2*M_PI;
        local_cluster_zeta[n][2] += sin(theta);
        local_cluster_xi[n][2] += cos(theta);
      }
      else
        local_cluster_xi[n][2] += latsites[k][2];
    }
  }

  // Find the global cluster size for each cluster across all processes
  delete [] cluster_size;
  delete [] cluster_nsites;
  cluster_size = new int[noccupied] ();
  cluster_nsites = new int[noccupied] ();
  MPI_Allreduce (local_cluster_size, cluster_size, noccupied, MPI_INT,
    MPI_SUM, world);
  MPI_Allreduce (local_cluster_nsites, cluster_nsites, noccupied, MPI_INT,
    MPI_SUM, world);
  delete [] local_cluster_nsites;
  delete [] local_cluster_size;

  // Now average the center in the complex plane <z>_a = (<xi>,<zeta>) across
  // all processes for all clusters
  double** cluster_xi;
  double** cluster_zeta;
  memory->create (cluster_xi, noccupied, 3, "ComputeFrenkel:cluster_xi");
  memory->create (cluster_zeta, noccupied, 3, "ComputeFrenkel:cluster_zeta");
  for ( int i = 0; i < noccupied * 3; i++ )
    cluster_xi[0][i] = cluster_zeta[0][i] = 0.0;
  MPI_Allreduce (local_cluster_xi[0], cluster_xi[0], noccupied*3, MPI_DOUBLE,
    MPI_SUM, world);
  MPI_Allreduce (local_cluster_zeta[0], cluster_zeta[0], noccupied*3,
    MPI_DOUBLE, MPI_SUM, world);
  memory->destroy (local_cluster_xi);
  memory->destroy (local_cluster_zeta);

  // If not periodic in that direction, then just divide xi by the cluster
  // size to get the center; else back out the angle and the center
  double** cluster_approx_center;
  memory->create (cluster_approx_center, noccupied, 3,
    "ComputeFrenkel::cluster_approx_center");
  for ( int n = 0; n < noccupied; n++ ) {
    if ( cluster_nsites[n] == 0 ) continue;
    cluster_xi[n][0] /= cluster_nsites[n];
    cluster_xi[n][1] /= cluster_nsites[n];
    cluster_xi[n][2] /= cluster_nsites[n];
    cluster_zeta[n][0] /= cluster_nsites[n];
    cluster_zeta[n][1] /= cluster_nsites[n];
    cluster_zeta[n][2] /= cluster_nsites[n];
    if ( domain->xperiodic ) {
      theta = atan2(-cluster_zeta[n][0], -cluster_xi[n][0]) + M_PI;
      cluster_approx_center[n][0] = domain->xprd * theta / (2.0*M_PI)
        + domain->boxlo[0];
    }
    else
      cluster_approx_center[n][0] = cluster_xi[n][0];
    if ( domain->yperiodic ) {
      theta = atan2(-cluster_zeta[n][1], -cluster_xi[n][1]) + M_PI;
      cluster_approx_center[n][1] = domain->yprd * theta / (2.0*M_PI)
        + domain->boxlo[1];
    }
    else
      cluster_approx_center[n][1] = cluster_xi[n][1];
    if ( domain->zperiodic ) {
      theta = atan2(-cluster_zeta[n][2], -cluster_xi[n][2]) + M_PI;
      cluster_approx_center[n][2] = domain->zprd * theta / (2.0*M_PI)
        + domain->boxlo[2];
    }
    else
      cluster_approx_center[n][2] = cluster_xi[n][2];

    domain->remap (cluster_approx_center[n]);

  }
  memory->destroy (cluster_zeta);
  memory->destroy (cluster_xi);

  // Now find the center using the approximate center to pick the right image
  double** local_cluster_x;
  double nearest_image[3];
  memory->create (local_cluster_x, noccupied, 3,
    "ComputeFrenkel:local_cluster_x");
  for ( int i = 0; i < noccupied * 3; i++ )
    local_cluster_x[0][i] = 0.0;
  memory->destroy (cluster_center);
  memory->create (cluster_center, noccupied, 3,
    "ComputeFrenkel::cluster_center");
  for ( int i = 0; i < noccupied * 3; i++ )
    cluster_center[0][i] = 0.0;
  for ( int k = 0; k < nlatsites; k++ ) {
    if ( clusterID[k] == 0 ) continue;
    n = clusterID2occupied_index (clusterID[k]);
    domain->closest_image (cluster_approx_center[n], latsites[k],
      nearest_image);
    local_cluster_x[n][0] += nearest_image[0] / cluster_nsites[n];
    local_cluster_x[n][1] += nearest_image[1] / cluster_nsites[n];
    local_cluster_x[n][2] += nearest_image[2] / cluster_nsites[n];
  }
  MPI_Allreduce (local_cluster_x[0], cluster_center[0], noccupied * 3,
    MPI_DOUBLE, MPI_SUM, world);

  memory->destroy (local_cluster_x);
  memory->destroy (cluster_approx_center);

}

/****************************************************************************/

void ComputeFrenkel::find_occupied_clusters () { // {{{1

  tagint* local_occupied_cluster_ID;
  int local_noccupied = 0;
  for ( int k = 0; k < nlatsites; k++ )
    if ( clusterID[k] > 0 ) local_noccupied += 1;
  memory->create (local_occupied_cluster_ID, local_noccupied,
    "ComputeFrenkel:occupied_cluster_ID");

  // Assign occupied clusters to an array
  int n = 0;
  for ( int k = 0; k < nlatsites; k++ )
    if ( clusterID[k] > 0 ) local_occupied_cluster_ID[n++] = clusterID[k];
  // sort, then remove duplicates (defects with more than one site involved)
  qsort (local_occupied_cluster_ID, local_noccupied,
    sizeof(local_occupied_cluster_ID[0]), &compareIDs);
  tagint previous = ( local_noccupied > 0 ? local_occupied_cluster_ID[0] : 0 );
  for ( n = 1; n < local_noccupied; n++ ) {
    if ( previous <= 0 ) break;
    // If previous entry is the same as this one, delete this entry
    while ( local_occupied_cluster_ID[n] == previous ) {
      for ( int n1 = n; n1 < local_noccupied - 1; n1++ )
        local_occupied_cluster_ID[n1] = local_occupied_cluster_ID[n1+1];
      local_occupied_cluster_ID[local_noccupied-1] = -1;
    }
    previous = local_occupied_cluster_ID[n];
  }
  // Shrink array so it doesn't include negative entries
  for ( n = 0; n < local_noccupied; n++ )
    if ( local_occupied_cluster_ID[n] < 0 ) break;
  local_noccupied = n;
  memory->grow (local_occupied_cluster_ID, local_noccupied,
    "ComputeFrenkel:reallocate_local_occupied_cluster_ID");

  // Now make the array global (size will be number of clusters across ALL
  // processes)
  MPI_Allreduce (&local_noccupied, &noccupied, 1, MPI_INT, MPI_SUM, world);
  memory->destroy (occupied_cluster_ID);
  memory->create (occupied_cluster_ID, noccupied,
    "ComputeFrenkel:occupied_cluster_ID");
  int* nreceive = new int[comm->nprocs] ();
  MPI_Allgather (&local_noccupied, 1, MPI_INT, nreceive, 1, MPI_INT, world);
  int displ;
  MPI_Scan (&local_noccupied, &displ, 1, MPI_INT, MPI_SUM, world);
  displ -= local_noccupied;
  int* displacement = new int[comm->nprocs] ();
  MPI_Allgather (&displ, 1, MPI_INT, displacement, 1, MPI_INT, world);
  MPI_Allgatherv (local_occupied_cluster_ID, local_noccupied, MPI_LMP_TAGINT,
    occupied_cluster_ID, nreceive, displacement, MPI_LMP_TAGINT, world);
  delete [] nreceive;
  delete [] displacement;
  memory->destroy (local_occupied_cluster_ID);

  // sort the big array and once again remove duplicates
  qsort (occupied_cluster_ID, noccupied,
    sizeof(occupied_cluster_ID[0]), &compareIDs);
  previous = ( noccupied > 0 ? occupied_cluster_ID[0] : 0 );
  for ( n = 1; n < noccupied; n++ ) {
    if ( previous <= 0 ) break;
    // If previous entry is the same as this one, delete this entry
    while ( occupied_cluster_ID[n] == previous ) {
      for ( int n1 = n; n1 < noccupied - 1; n1++ )
        occupied_cluster_ID[n1] = occupied_cluster_ID[n1+1];
      occupied_cluster_ID[noccupied-1] = -1;
    }
    previous = occupied_cluster_ID[n];
  }
  for ( n = 0; n < noccupied; n++ )
    if ( occupied_cluster_ID[n] < 0 ) break;
  noccupied = n;
  memory->grow (occupied_cluster_ID, noccupied,
    "ComputeFrenkel:reallocate_occupied_cluster_ID");

}

/****************************************************************************/

int ComputeFrenkel::clusterID2occupied_index (int id) { // {{{1

  for ( int i = 0; i < noccupied; i++ )
    if ( id == occupied_cluster_ID[i] ) return i;
  return -1;

}

/****************************************************************************/

int ComputeFrenkel::compareIDs (const void* a, const void* b) { // {{{1
  return ( *(int*)a - *(int*)b );
}

/****************************************************************************/

void ComputeFrenkel::compute_vector () { // {{{1

  if ( invoked_vector == update->ntimestep ) return;
  invoked_vector = update->ntimestep;

  // If we haven't identified the defects yet, do so now
  if ( invoked_find_defects != update->ntimestep )
    find_defects();

  // Zero out the array
  for ( int n = 0; n < size_vector; n++ )
    vector[n] = 0.0;

  // Add up all sites with zero (vacancies), two (interstitials), and more
  // than two occupants (irregular).
  int nints = 0, nvacs = 0, nirreg = 0;
  for ( int k = 0; k < nlatsites; k++ ) {
    switch ( noccupants[k] ) {
      case 0:
        nvacs += 1;
        break;
      case 1:
        break;
      case 2:
        nints += 1;
        break;
      default:
        nints += 1;
        nirreg += 1;
    }
  }

  int nvacancies = 0, ninterstitials = 0, nirregular = 0;
  MPI_Allreduce (&nvacs, &nvacancies, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce (&nints, &ninterstitials, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce (&nirreg, &nirregular, 1, MPI_INT, MPI_SUM, world);

  vector[0] = static_cast<double>(nvacancies);
  vector[1] = static_cast<double>(ninterstitials);
  vector[2] = static_cast<double>(nirregular);

}

/****************************************************************************/

void ComputeFrenkel::compute_array () { // {{{1

  if ( invoked_array == update->ntimestep ) return;
  invoked_array = update->ntimestep;

  // If we haven't identified the defects or clusters yet, do so now
  if ( invoked_find_defects != update->ntimestep )
    find_defects();
  if ( invoked_find_clusters != update->ntimestep )
    find_clusters();

  // Initialize the array
  for ( int i = 0; i < size_array_cols * size_array_rows; i++ )
    array[0][i] = 0;

  // Add up # of -1's, -2's, etc. in the cluster size lists.
  for ( int n = 0; n < noccupied; n++ ) {
    if ( cluster_size[n] == 0 ) continue;
    if ( cluster_size[n] > 0 and cluster_size[n] < size_array_cols )
      array[1][cluster_size[n] - 1] += 1;
    else if ( cluster_size[n] >= size_array_cols )
      array[1][size_array_cols - 1] += 1;
    else if ( cluster_size[n] < 0 and cluster_size[n] > -size_array_cols )
      array[0][-cluster_size[n] - 1] += 1;
    else
      array[0][size_array_cols - 1] += 1;
  }

}

/****************************************************************************/

void ComputeFrenkel::compute_peratom () { // {{{1

  invoked_peratom = update->ntimestep;

  // If we haven't identified the defects yet, do so now
  if ( invoked_find_defects != update->ntimestep )
    find_defects();

  vector_atom = mindist;

}

/****************************************************************************/

void ComputeFrenkel::compute_local () { // {{{1

  invoked_local = update->ntimestep;

  // If we haven't identified defects yet, do that, too
  if ( invoked_find_defects != update->ntimestep )
    find_defects();

  // Ditto for clusters
  if ( invoked_find_clusters != update->ntimestep )
    find_clusters();

//  if ( noccupied == 0 ) return;

  // Find out whether the cluster belongs to this subdomain
  // (this prevents duplicate output)
  bool* owned = new bool[noccupied] ();
  int nowned = 0;
  for ( int i = 0; i < noccupied; i++ ) {
    if ( cluster_center[i][0] >= domain->sublo[0] and
         cluster_center[i][0] < domain->subhi[0] and
         cluster_center[i][1] >= domain->sublo[1] and
         cluster_center[i][1] < domain->subhi[1] and
         cluster_center[i][2] >= domain->sublo[2] and
         cluster_center[i][2] < domain->subhi[2] ) {
      owned[i] = true;
      nowned++;
    }
    else
      owned[i] = false;
  }

  size_local_rows = nowned;

  if ( nowned == 0 ) {
    delete [] owned;
    return;
  }

  memory->destroy (array_local);
  memory->create (array_local, size_local_rows, size_local_cols,
    "ComputeFrenkel:array_local");

  // Store the following quantities in the local array:
  // id, size, center_x, center_y, center_z
  int i = 0;
  for ( int j = 0; j < noccupied; j++ ) {
    if ( not owned[j] ) continue;
    array_local[i][0] = static_cast<double> (occupied_cluster_ID[j]);
    array_local[i][1] = static_cast<double> (cluster_size[j]);
/*    array_local[i][2] = (cluster_center[j][0] - domain->boxlo[0])/domain->xprd;
    array_local[i][3] = (cluster_center[j][1] - domain->boxlo[1])/domain->yprd;
    array_local[i][4] = (cluster_center[j][2] - domain->boxlo[2])/domain->zprd;
*/
    array_local[i][2] = cluster_center[j][0];
    array_local[i][3] = cluster_center[j][1];
    array_local[i][4] = cluster_center[j][2];
    i++;
  }
  delete [] owned;

}

/****************************************************************************/

void ComputeFrenkel::create_lattice_sites () { // {{{1

  // First, isolate the current atoms in their own group
  int *flags = new int[atom->nlocal];
  for (int m = 0; m < atom->nlocal; m++)
    flags[m] = 1;
  char *createstring = new char[strlen(Frenkel_everything)+1];
  strcpy (createstring,Frenkel_everything);
  group->create (createstring,flags);
  delete [] flags;
  delete [] createstring;

  // Now create the lattice points; entities created here are FAKE sites
  if ( domain->lattice == nullptr )
     error->all (FLERR, "Use of compute style frenkel with undefined lattice");
  if ( sitefile ) {
    // User provided a file that creates atoms in arbitrary fashion; parse it!
    const int maxline = 1024;
    char line[maxline]; // Assume no user is stupid enough to write looong lines
    char wholeline[maxline];
    char* loc;
    FILE* sfile;
    strcpy (wholeline,"");
    sfile = fopen (sitefile, "r");
    turnoffoutput();
    while ( true ) {
      fgets (line, maxline, sfile);
      if ( feof(sfile) ) break;
      // Remove end of line character
      loc = strrchr (line, '\n');
      if ( loc != nullptr ) loc[0] = 0;
      // Join line to the next one(s), if necessary
      loc = strrchr (line, '&');
      if ( loc != nullptr and loc[1] == '\0' ) {
        loc[0] = '\0';
        strcat (wholeline, line);
        continue;
      }
      strcat (wholeline,line);
      input->one (wholeline);
      strcpy (wholeline,"");
    }
    revertoutput();

  }
  //else if ( iregion == -1 ) {
  else if ( not region ) {
    // do equivalent of "create_atoms 1 box"
    char **createcommand = new char*[2];
    createcommand[0] = new char[2];
    createcommand[1] = new char[4];
    strcpy(createcommand[0],"1");
    strcpy(createcommand[1],"box");
    CreateAtoms create_latticepoints(lmp);
    turnoffoutput();
    create_latticepoints.command(2,createcommand);
    revertoutput();
    delete [] createcommand[1];
    delete [] createcommand[0];
    delete [] createcommand;
  }
  else {
    // do equivalent of "create_atoms 1 region mbox"
    char **createcommand = new char*[3];
    createcommand[0] = new char[2];
    createcommand[1] = new char[7];
    strcpy(createcommand[0],"1");
    strcpy(createcommand[1],"region");
    //createcommand[2] = idregion;
    createcommand[2] = region->id;
    CreateAtoms create_latticepoints(lmp);
    turnoffoutput();
    create_latticepoints.command(3,createcommand);
    revertoutput();
    delete [] createcommand[1];
    delete [] createcommand[0];
    delete [] createcommand;
  }

  // Now set flags so we can put the lattice points in their own group
  int everything = group->find(Frenkel_everything);
  if ( everything == -1 ) {
    char errorstring[80];
    sprintf(errorstring,"Failed to find group %s in compute style frenkel",
      Frenkel_everything);
    error->all(FLERR, errorstring);
  }
  flags = new int[atom->nlocal];
  for (int m = 0; m < atom->nlocal; m++) {
    if ( atom->mask[m] & group->bitmask[everything] )
      flags[m] = 0; // regular atoms
    else
      flags[m] = 1; // lattice sites
  }

  // Put the lattice points in their own group
  createstring = new char[strlen(Frenkel_latticesites)+1];
  strcpy(createstring,Frenkel_latticesites);
  turnoffoutput();
  group->create(createstring,flags);
  revertoutput();
  delete [] flags;
  delete [] createstring;
  int latticesitesgroup = group->find(Frenkel_latticesites);
  if ( latticesitesgroup == -1 )
    error->all(FLERR,
      "Unable to find lattice sites group for dump style frenkel");

  // Set up lattice sites array (we don't know the right size yet)
  int nsites = group->count(latticesitesgroup);
  if ( nsites == 0 ) error->warning (FLERR, "I didn't find any lattice sites");
  double **sites =
    memory->create (sites, nsites, 3, "Frenkel-temp-lattice-sites");
  int n = 0;
  for (int m = 0; m < atom->nlocal; m++) {
     if ( atom->mask[m] & group->bitmask[latticesitesgroup] ) {
        sites[n][0] = atom->x[m][0];
        sites[n][1] = atom->x[m][1];
        sites[n][2] = atom->x[m][2];
        n++;
     }
  }
  nlatsites = n;  // THIS is the right size
  nlatghosts = 0;
  memory->destroy (latsites);
  memory->create (latsites, nlatsites, 3, "Frenkel-lattice-sites");
  for ( n = 0; n < nlatsites; n++ ) {
     latsites[n][0] = sites[n][0];
     latsites[n][1] = sites[n][1];
     latsites[n][2] = sites[n][2];
  }
  memory->destroy(sites);

  // Delete the atoms that you used to place the lattice sites
  char **deletecommand = new char*[4];
  deletecommand[0] = new char[6];
  deletecommand[1] = new char[strlen(Frenkel_latticesites)+1];
  deletecommand[2] = new char[strlen("compress")+1];
  deletecommand[3] = new char[strlen("no")+1];
  strcpy(deletecommand[0],"group");
  strcpy(deletecommand[1],Frenkel_latticesites);
  DeleteAtoms delete_vacancies(lmp);
  turnoffoutput();
  delete_vacancies.command(2,deletecommand);
  revertoutput();
  delete [] deletecommand[3];
  delete [] deletecommand[2];
  delete [] deletecommand[1];
  delete [] deletecommand[0];
  delete [] deletecommand;

  // Delete the group IDs as well
  char **groupcommand = new char*[2];
  groupcommand[0] = new char[strlen(Frenkel_latticesites)+1];
  groupcommand[1] = new char[7];
  strcpy(groupcommand[0],Frenkel_latticesites);
  strcpy(groupcommand[1],"delete");
  turnoffoutput();
  group->assign(2,groupcommand);
  revertoutput();
  delete [] groupcommand[0];
  groupcommand[0] = new char[strlen(Frenkel_everything)+1];
  strcpy(groupcommand[0],Frenkel_everything);
  turnoffoutput();
  group->assign(2,groupcommand);
  revertoutput();
  delete [] groupcommand[1];
  delete [] groupcommand[0];
  delete [] groupcommand;

  // Now give each lattice site a unique ID ("tag")
  memory->destroy (site_tag);
  memory->create (site_tag, nlatsites, "ComputeFrenkel:site_tag");
  first_local_tag = 0;
  tagint nlats = nlatsites;
  MPI_Scan (&nlats, &first_local_tag, 1, MPI_LMP_TAGINT, MPI_SUM, world);
  first_local_tag = first_local_tag - nlatsites + 1 + atom->natoms;
  // The starting tag is now natoms + 1 on process 0,
  //  nlatsites[proc 0] + natoms + 1 on process 1, etc.
  for ( int k = 0; k < nlatsites; k++ )
    site_tag[k] = first_local_tag + k;

  // Make a copy of the lattice sites' coordinates if we're rescaling
  memory->destroy (latsites0);
  if ( rescale ) {
    memory->create (latsites0, nlatsites, 3, "ComputeFrenkel:latsites0");
    memcpy (*latsites0, *latsites, nlatsites * 3 * sizeof(double));
  }
  else
    latsites0 = nullptr;

}

/****************************************************************************/

int ComputeFrenkel::site_tag2index (tagint tag) { // {{{1

  int idx = static_cast<int> (tag - this->first_local_tag);
  if ( idx >= 0 and idx < nlatsites )
    return idx;

  // Tag belongs only to a ghost; find it!
  for ( int i = nlatsites; i < nlatsites + nlatghosts; i++ )
    if ( site_tag[i] == tag )
      return i;

  // Hmm...this tag doesn't seem to exist!  Why are you looking for it?
fprintf(stderr, "WARNING (proc %d): Didn't find an index for tag %d (%s, line %d).\n", comm->me, tag, FLERR);
  return -1;

}

/****************************************************************************/

void ComputeFrenkel::exchange_lattice_ghosts () { // {{{1

  // Exchange sites within one cutoff distance of the edge of the box with
  // adjacent processes.  Quantities exchanged:  site tag, site coordinates,
  // cluster ID's, and site occupancies.

  tagint** tag_send;
  tagint** tag_recv;
  tagint** clusterID_send;
  tagint** clusterID_recv;
  double*** x_send;
  double*** x_recv;
  int** occup_send;
  int** occup_recv;
  int* n_send;
  int* n_recv;
  const int NINCR = 100;
  int max_size = NINCR;
  int* idx;

  nlatghosts = 0;
  n_send = new int[comm->nprocs];
  n_recv = new int[comm->nprocs];
  idx = new int[comm->nprocs];
  memory->create (tag_send, comm->nprocs, max_size,
    "ComputeFrenkel:tag_send");
  memory->create (x_send, comm->nprocs, max_size, 3,
    "ComputeFrenkel:x_send");
  memory->create (occup_send, comm->nprocs, max_size,
    "ComputeFrenkel:occup_send");
  memory->create (clusterID_send, comm->nprocs, max_size,
    "ComputeFrenkel:clusterID_send");
  for ( int p = 0; p < comm->nprocs; p++ )
    idx[p] = n_send[p] = n_recv[p] = 0;
  for ( int i = 0; i < comm->nprocs * max_size; i++ )
    tag_send[0][i] = -1;
  for ( int i = 0; i < comm->nprocs * max_size * 3; i++ )
    x_send[0][0][i] = 0.0;
  for ( int i = 0; i < comm->nprocs * max_size; i++ )
    occup_send[0][i] = 0;
  for ( int i = 0; i < comm->nprocs * max_size; i++ )
    clusterID_send[0][i] = 0;

  // Find out which sites are near the boundaries
  // and which processes those boundaries correspond to.
  double dr[2][3];
  bool* already_sent = new bool[comm->nprocs];
  for ( int k = 0; k < nlatsites; k++ ) {
    dr[0][0] = latsites[k][0] - domain->sublo[0];
    dr[0][1] = latsites[k][1] - domain->sublo[1];
    dr[0][2] = latsites[k][2] - domain->sublo[2];
    dr[1][0] = domain->subhi[0] - latsites[k][0];
    dr[1][1] = domain->subhi[1] - latsites[k][1];
    dr[1][2] = domain->subhi[2] - latsites[k][2];

    // Prepare list of processes to which we've sent this site to (to avoid
    // duplication)
    for ( int p = 0; p < comm->nprocs; p++ )
      already_sent[p] = false;

    // Test processes in all 27 directions from this one
    for ( int ix = -1; ix <= 1; ix++ ) {
      // If at the end of the domain, ignore those directions unless
      // box is periodic in that direction
      if ( ix == -1 and comm->myloc[0] == 0 and not domain->xperiodic )
        continue;
      if ( ix == 1 and comm->myloc[0] == comm->procgrid[0] - 1 and
            not domain->xperiodic )
        continue;
      if ( ix == -1 and dr[0][0] > cutoff + SMALL ) continue;
      if ( ix == 1 and dr[1][0] > cutoff + SMALL ) continue;
      for ( int iy = -1; iy <= 1; iy++ ) {
        if ( iy == -1 and comm->myloc[1] == 0 and not domain->yperiodic )
          continue;
        if ( iy == 1 and comm->myloc[1] == comm->procgrid[1] - 1 and
            not domain->yperiodic )
          continue;
        if ( iy == -1 and dr[0][1] > cutoff + SMALL ) continue;
        if ( iy == 1 and dr[1][1] > cutoff + SMALL ) continue;
        for ( int iz = -1; iz <= 1; iz++ ) {
          if ( ix == 0 and iy == 0 and iz == 0 ) continue;
          if ( iz == -1 and comm->myloc[2] == 0 and not domain->zperiodic )
            continue;
          if ( iz == 1 and comm->myloc[2] == comm->procgrid[2] - 1 and
              not domain->zperiodic )
            continue;
          if ( iz == -1 and dr[0][2] > cutoff + SMALL ) continue;
          if ( iz == 1 and dr[1][2] > cutoff + SMALL ) continue;
          // Should only get here when exchanging with an adjacent process
          int p = process_neighbor (ix, iy, iz);
          // Don't add site to process multiple times
          if ( already_sent[p] ) continue;
          n_send[p] += 1;
          if ( n_send[p] > max_size ) { // Grow arrays if necessary
            this->reallocate_array (tag_send, comm->nprocs, max_size,
              comm->nprocs, max_size + NINCR);
            this->reallocate_array (x_send, comm->nprocs, max_size, 3,
              comm->nprocs, max_size + NINCR, 3);
            this->reallocate_array (occup_send, comm->nprocs, max_size,
              comm->nprocs, max_size + NINCR);
            this->reallocate_array (clusterID_send, comm->nprocs, max_size,
              comm->nprocs, max_size + NINCR);
            max_size += NINCR;
          }
          tag_send[p][idx[p]] = site_tag[k];
          occup_send[p][idx[p]] = noccupants[k];
          if ( clusterID ) clusterID_send[p][idx[p]] = clusterID[k];
          x_send[p][idx[p]][0] = latsites[k][0];
          x_send[p][idx[p]][1] = latsites[k][1];
          x_send[p][idx[p]][2] = latsites[k][2];
          idx[p] += 1;
          already_sent[p] = true;
        }
      }
    }
  }
  delete [] already_sent;

  // Send your sites to them and get theirs in return
  MPI_Request *send_request = new MPI_Request[comm->nprocs] ();
  MPI_Request *recv_request = new MPI_Request[comm->nprocs] ();
  send_request[comm->me] = MPI_REQUEST_NULL;
  recv_request[comm->me] = MPI_REQUEST_NULL;
  for ( int p = 0; p < comm->nprocs; p++ ) {
    if ( p == comm->me )
      n_recv[p] = n_send[p];
    else {
      MPI_Isend (&n_send[p], 1, MPI_INT, p, comm->me, world, &send_request[p]);
      MPI_Irecv (&n_recv[p], 1, MPI_INT, p, p, world, &recv_request[p]);
    }
  }
  MPI_Status* send_status = new MPI_Status[comm->nprocs] ();
  MPI_Status* recv_status = new MPI_Status[comm->nprocs]();
//  for ( int i = 0; i < comm->nprocs; i++ )
//    send_status[i] = recv_status[i] = MPI_Status();
  if ( comm->nprocs > 1 ) {
    MPI_Waitall (comm->nprocs, send_request, send_status);
    MPI_Waitall (comm->nprocs, recv_request, recv_status);
  }

  max_size = 0;
  for ( int p = 0; p < comm->nprocs; p++ )
    max_size = MAX (max_size, n_recv[p]);
  if ( max_size == 0 ) {
    memory->destroy (tag_send);
    memory->destroy (x_send);
    memory->destroy (occup_send);
    memory->destroy (clusterID_send);
    delete [] n_send;
    delete [] n_recv;
    delete [] idx;
    delete [] send_request;
    delete [] recv_request;
    delete [] send_status;
    delete [] recv_status;
    return; // Should only happen if regions don't cross domain boundaries
  }
  memory->create (tag_recv, comm->nprocs, max_size, "ComputeFrenkel:tag_recv");
  memory->create (x_recv, comm->nprocs, max_size, 3, "ComputeFrenkel:x_recv");
  memory->create (occup_recv, comm->nprocs, max_size,
    "ComputeFrenkel:occup_recv");
  memory->create (clusterID_recv, comm->nprocs, max_size,
    "ComputeFrenkel:clusterID_recv");
  for ( int i = 0; i < comm->nprocs * max_size; i++ )
    tag_recv[0][i] = 0;
  for ( int i = 0; i < comm->nprocs * max_size; i++ )
    occup_recv[0][i] = 0;
  for ( int i = 0; i < comm->nprocs * max_size * 3; i++ )
    x_recv[0][0][i] = 0.0;
  for ( int i = 0; i < comm->nprocs * max_size; i++ )
    clusterID_recv[0][i] = 0;
  MPI_Request* send_request_x = new MPI_Request[comm->nprocs] ();
  MPI_Request* recv_request_x = new MPI_Request[comm->nprocs] ();
  MPI_Request* send_request_o = new MPI_Request[comm->nprocs] ();
  MPI_Request* recv_request_o = new MPI_Request[comm->nprocs] ();
  MPI_Request* send_request_c = new MPI_Request[comm->nprocs] ();
  MPI_Request* recv_request_c = new MPI_Request[comm->nprocs] ();
  MPI_Status* send_status_x = new MPI_Status[comm->nprocs] ();
  MPI_Status* recv_status_x = new MPI_Status[comm->nprocs] ();
  MPI_Status* send_status_o = new MPI_Status[comm->nprocs] ();
  MPI_Status* recv_status_o = new MPI_Status[comm->nprocs] ();
  MPI_Status* send_status_c = new MPI_Status[comm->nprocs] ();
  MPI_Status* recv_status_c = new MPI_Status[comm->nprocs] ();
  send_request[comm->me] = MPI_REQUEST_NULL;
  recv_request[comm->me] = MPI_REQUEST_NULL;
  send_request_x[comm->me] = MPI_REQUEST_NULL;
  recv_request_x[comm->me] = MPI_REQUEST_NULL;
  send_request_o[comm->me] = MPI_REQUEST_NULL;
  recv_request_o[comm->me] = MPI_REQUEST_NULL;
  send_request_c[comm->me] = MPI_REQUEST_NULL;
  recv_request_c[comm->me] = MPI_REQUEST_NULL;
  for ( int p = 0; p < comm->nprocs; p++ ) {
    if ( p == comm->me ) { // Don't talk to yourself, just copy
      for ( int i = 0; i < n_send[p]; i++ ) {
        tag_recv[p][i] = tag_send[p][i];
        occup_recv[p][i] = occup_send[p][i]; // FIXME - which one should this be?
        //occup_recv[p][i] += occup_send[p][i];
        clusterID_recv[p][i] = clusterID_send[p][i];
        x_recv[p][i][0] = x_send[p][i][0];
        x_recv[p][i][1] = x_send[p][i][1];
        x_recv[p][i][2] = x_send[p][i][2];
      }
      continue;
    }
    // First send...
    if ( n_send[p] > 0 ) {
      MPI_Isend (tag_send[p], n_send[p], MPI_LMP_TAGINT, p,
          comm->me, world, &send_request[p]);
      MPI_Isend (x_send[p][0], n_send[p] * 3, MPI_DOUBLE, p,
          comm->me + comm->nprocs, world, &send_request_x[p]);
      MPI_Isend (occup_send[p], n_send[p], MPI_INT, p,
          comm->me + comm->nprocs * 2, world, &send_request_o[p]);
      MPI_Isend (clusterID_send[p], n_send[p], MPI_LMP_TAGINT, p,
          comm->me + comm->nprocs * 3, world, &send_request_c[p]);
    }
    else {
      send_request[p] = MPI_REQUEST_NULL;
      send_request_x[p] = MPI_REQUEST_NULL;
      send_request_o[p] = MPI_REQUEST_NULL;
      send_request_c[p] = MPI_REQUEST_NULL;
    }
    // ...now receive
    if ( n_recv[p] > 0 ) {
      MPI_Irecv (tag_recv[p], n_recv[p], MPI_LMP_TAGINT, p, p,
        world, &recv_request[p]);
      MPI_Irecv (x_recv[p][0], n_recv[p] * 3, MPI_DOUBLE, p, p + comm->nprocs,
        world, &recv_request_x[p]);
      MPI_Irecv (occup_recv[p], n_recv[p], MPI_INT, p, p + comm->nprocs * 2,
        world, &recv_request_o[p]);
      MPI_Irecv (clusterID_recv[p], n_recv[p], MPI_LMP_TAGINT, p,
        p + comm->nprocs * 3, world, &recv_request_c[p]);
    }
    else {
      recv_request[p] = MPI_REQUEST_NULL;
      recv_request_x[p] = MPI_REQUEST_NULL;
      recv_request_o[p] = MPI_REQUEST_NULL;
      recv_request_c[p] = MPI_REQUEST_NULL;
    }
  }
  if ( comm->nprocs > 1 ) {
    MPI_Waitall (comm->nprocs, send_request, send_status);
    MPI_Waitall (comm->nprocs, recv_request, recv_status);
    MPI_Waitall (comm->nprocs, send_request_x, send_status_x);
    MPI_Waitall (comm->nprocs, recv_request_x, recv_status_x);
    MPI_Waitall (comm->nprocs, send_request_o, send_status_o);
    MPI_Waitall (comm->nprocs, recv_request_o, recv_status_o);
    MPI_Waitall (comm->nprocs, send_request_c, send_status_c);
    MPI_Waitall (comm->nprocs, recv_request_c, recv_status_c);
  }

  // Reallocate the necessary memory
  nlatghosts = 0;
  for ( int p = 0; p < comm->nprocs; p++ )
    nlatghosts += n_recv[p];
  memory->grow (latsites, nlatsites + nlatghosts, 3, "ComputeFrenkel:sites2");
  memory->grow (site_tag, nlatsites + nlatghosts, "ComputeFrenkel:site_tag2");
  memory->grow (noccupants, nlatsites + nlatghosts,
    "ComputeFrenkel:noccupants2");
  if ( clusterID ) memory->grow (clusterID, nlatsites + nlatghosts,
    "ComputeFrenkel:clusterID2");

  // Update latsites, site_tag, clusterID, and nlatghosts
  int kk = nlatsites;
  for ( int p = 0; p < comm->nprocs; p++ ) {
    for ( int i = 0; i < n_recv[p]; i++ ) {
      site_tag[kk] = tag_recv[p][i];
      latsites[kk][0] = x_recv[p][i][0];
      latsites[kk][1] = x_recv[p][i][1];
      latsites[kk][2] = x_recv[p][i][2];
      noccupants[kk] = occup_recv[p][i];
      if ( clusterID ) clusterID[kk] = clusterID_recv[p][i];
      kk++;
    }
  }

  // Clean up
  memory->destroy (tag_send);
  memory->destroy (x_send);
  memory->destroy (occup_send);
  memory->destroy (clusterID_send);
  memory->destroy (tag_recv);
  memory->destroy (x_recv);
  memory->destroy (occup_recv);
  memory->destroy (clusterID_recv);
  delete [] n_send;
  delete [] n_recv;
  delete [] idx;
  delete [] send_request;
  delete [] recv_request;
  delete [] send_request_x;
  delete [] recv_request_x;
  delete [] send_request_o;
  delete [] recv_request_o;
  delete [] send_request_c;
  delete [] recv_request_c;
  delete [] send_status;
  delete [] recv_status;
  delete [] send_status_x;
  delete [] recv_status_x;
  delete [] send_status_o;
  delete [] recv_status_o;
  delete [] send_status_c;
  delete [] recv_status_c;

}

/****************************************************************************/

void ComputeFrenkel::construct_site_nlists () { // {{{1

  // Builds neighbor lists for every site in the subdomain

  double dx, dy, dz, drsq;
  double cutsq = cutoff * cutoff;
  int ix, iy, iz, i, j, k;
  delete [] nlist;
  nlist = new list<tagint> [nlatsites + nlatghosts] ();
  for ( int n = 0; n < nlatsites + nlatghosts; n++ ) {
    // Loop over all sites in this site's bin and those adjacent
    find_closest_bin (latsites[n], i, j, k);
    for ( int ii = i - 1; ii <= i + 1; ii++ ) {
      ix = ii;
      for ( int jj = j - 1; jj <= j + 1; jj++ ) {
        iy = jj;
        for ( int kk = k - 1; kk <= k + 1; kk++ ) {
          iz = kk;
          bin_pbc (ix, iy, iz);
          for ( int l = 0; l < nlatbins[3]; l++ ) {
            int m = latbins[ix][iy][iz][l];
            if ( m < 0 ) break;
            if ( m == n ) continue;
            dx = latsites[n][0] - latsites[m][0];
            dy = latsites[n][1] - latsites[m][1];
            dz = latsites[n][2] - latsites[m][2];
            domain->minimum_image (dx, dy, dz);
            drsq = dx*dx + dy*dy + dz*dz;
            if ( drsq <= cutsq ) {
              // Add site m to neighbor list of site n
              nlist[n].push_back (site_tag[m]);
            }
          }
        }
      }
    }
    // Sort neighbor list and remove duplicates
    nlist[n].sort();
    nlist[n].unique();

  }

}

/****************************************************************************/

void ComputeFrenkel::put_sites_in_bins () { // {{{1

  // Each site is assigned to a unique bin of width cutoff.
  nlatbins[0] = ceil((domain->subhi[0] - domain->sublo[0]) / binwidth);
  nlatbins[1] = ceil((domain->subhi[1] - domain->sublo[1]) / binwidth);
  nlatbins[2] = ceil((domain->subhi[2] - domain->sublo[2]) / binwidth);
  nlatbins[3] = domain->lattice->nbasis * BIN_GROW_SIZE; // can grow

  // Temporary array to store the index we're currently on
  int*** binindex = memory->create (binindex, nlatbins[0], nlatbins[1],
    nlatbins[2], "ComputeFrenkel:binindex");

  // Create the bins
  memory->destroy (latbins);
  memory->create (latbins, nlatbins[0], nlatbins[1], nlatbins[2], nlatbins[3],
    "ComputeFrenkel:nlatbins");

  // Initialize the bin contents
  int *iptr = binindex[0][0];
  for ( int i = 0; i < nlatbins[0]*nlatbins[1]*nlatbins[2]; i++ )
    iptr[i] = 0;
  iptr = latbins[0][0][0];
  for ( int i = 0; i < nlatbins[0]*nlatbins[1]*nlatbins[2]*nlatbins[3]; i++ )
    iptr[i] = -1;

  // Store indices of all lattice points in appropriate bins
  for ( int n = 0; n < nlatsites + nlatghosts; n++ ) {
    int i, j, k, l;
    find_closest_bin (latsites[n], i, j, k);
    l = binindex[i][j][k];
    latbins[i][j][k][l] = n;
    binindex[i][j][k] += 1;
    if ( binindex[i][j][k] >= nlatbins[3] ) {
      int startval = nlatbins[3];
      nlatbins[3] += BIN_GROW_SIZE;
      this->grow (latbins, nlatbins[0], nlatbins[1], nlatbins[2],
        nlatbins[3], "ComputeFrenkel:realloc-latbins");
      for ( int ii = 0; ii < nlatbins[0]; ii++ )
        for ( int jj = 0; jj < nlatbins[1]; jj++ )
          for ( int kk = 0; kk < nlatbins[2]; kk++ )
            for ( int ll = startval; ll < nlatbins[3]; ll++ )
              latbins[ii][jj][kk][ll] = -1;
    }
  }

  // Deallocate our counting array
  memory->destroy (binindex);

}

/****************************************************************************/

// void ComputeFrenkel::find_closest_bin {{{1
void ComputeFrenkel::find_closest_bin (double* r, int& i, int& j, int& k) {

  double x = r[0] - domain->sublo[0];
  double y = r[1] - domain->sublo[1];
  double z = r[2] - domain->sublo[2];
  i = nint(x/binwidth);
  j = nint(y/binwidth);
  k = nint(z/binwidth);

  // Apply periodic boundaries, if appropriate
  if ( domain->xperiodic and comm->procneigh[0][0] == comm->me )
    i = (nlatbins[0] + i) % nlatbins[0];
  else {
    i = MIN(i, nlatbins[0]-1);
    i = MAX(i, 0);
  }
  if ( domain->yperiodic and comm->procneigh[1][0] == comm->me )
    j = (nlatbins[1] + j) % nlatbins[1];
  else {
    j = MIN(j, nlatbins[1]-1);
    j = MAX(j, 0);
  }
  if ( domain->zperiodic and comm->procneigh[2][0] == comm->me )
    k = (nlatbins[2] + k) % nlatbins[2];
  else {
    k = MIN(k, nlatbins[2]-1);
    k = MAX(k, 0);
  }
  // (i,j,k) is now the bin on this process whose center is closest to r

}

/****************************************************************************/

void ComputeFrenkel::bin_pbc (int& i, int& j, int& k) {

  if ( i < 0 ) {
    if ( domain->xperiodic and comm->procneigh[0][0] == comm->me )
      i = nlatbins[0] - 1;
    else
      i = 0;
  }
  else if ( i >= nlatbins[0] ) {
    if ( domain->xperiodic and comm->procneigh[0][1] == comm->me )
      i = 0;
    else
      i = nlatbins[0] - 1;
  }
  if ( j < 0 ) {
    if ( domain->yperiodic and comm->procneigh[1][0] == comm->me )
      j = nlatbins[1] - 1;
    else
      j = 0;
  }
  else if ( j >= nlatbins[1] ) {
    if ( domain->yperiodic and comm->procneigh[1][1] == comm->me )
      j = 0;
    else
      j = nlatbins[1] - 1;
  }
  if ( k < 0 ) {
    if ( domain->zperiodic and comm->procneigh[2][0] == comm->me )
      k = nlatbins[2] - 1;
    else
      k = 0;
  }
  else if ( k >= nlatbins[2] ) {
    if ( domain->zperiodic and comm->procneigh[2][1] == comm->me )
      k = 0;
    else
      k = nlatbins[2] - 1;
  }

}

/****************************************************************************/

void ComputeFrenkel::construct_WS_cell () { // {{{1
    
  invoked_construct_WS_cell = update->ntimestep;
  
  // Constructs the Wigner-Seitz cell (in lattice units) from the basis
  // vectors and lattice directions.
  
  nnormal = 27 * domain->lattice->nbasis - 1;
  memory->destroy (normal);
  memory->create (normal, nnormal, 3, "ComputeFrenkel:normal");
  double **basis = domain->lattice->basis;
  double *a1 = domain->lattice->a1;
  double *a2 = domain->lattice->a2;
  double *a3 = domain->lattice->a3;

  for ( int i = 0; i < nnormal * 3; i++ )
    normal[0][i] = 0.0;

  int n = 0;
  for ( int j = 0; j < domain->lattice->nbasis; j++ )
    for ( int n1 = -1; n1 <= 1; n1++ )
      for ( int n2 = -1; n2 <= 1; n2++ )
        for ( int n3 = -1; n3 <= 1; n3++ ) {
          // Skip basis vector zero; that's the origin!
          if ( j == 0 and n1 == 0 and n2 == 0 and n3 == 0 ) continue;
          // The normal vector is a vector passing through lattice point
          // zero (arbitrary) and another lattice point either in this unit
          // cell or in an adjacent one.  The factor of 0.5 means it is also
          // the midpoint of that line segment.
          normal[n][0] = 0.5 * (basis[j][0]
            + n1*a1[0] + n2*a2[0] + n3*a3[0] - basis[0][0]);
          normal[n][1] = 0.5 * (basis[j][1]
            + n1*a1[1] + n2*a2[1] + n3*a3[1] - basis[0][1]);
          normal[n][2] = 0.5 * (basis[j][2]
            + n1*a1[2] + n2*a2[2] + n3*a3[2] - basis[0][2]);
          n = n + 1;
        }

}

/****************************************************************************/

bool ComputeFrenkel::inside_WS_cell (int n, int k) { // {{{1
  // Returns true if atom n is inside Wigner-Seitz cell k, false if not
  // This check is not necessary for atoms not near the edge of a process
  // domain; we COULD check that condition and speed this up.  However, we
  // would also have to check against (arbitrary) user-defined regions, so
  // that isn't as trivial as it sounds.

  // First, adjust the coordinates for periodic boundary conditions
  double r[3];
  domain->closest_image (latsites[k], atom->x[n], r);

  // Easy check:  if it's more than half a lattice unit from the center of the
  // Wigner-Seitz cell, it's outside the cell!
  if ( fabs(r[0] - latsites[k][0]) > 0.5 * domain->lattice->xlattice )
    return false;
  if ( fabs(r[1] - latsites[k][1]) > 0.5 * domain->lattice->ylattice )
    return false;
  if ( fabs(r[2] - latsites[k][2]) > 0.5 * domain->lattice->zlattice )
    return false;

  // If we haven't constructed the W-S cell, do it now
  if ( invoked_construct_WS_cell != update->ntimestep )
    construct_WS_cell();

  // It's inside the unit cell drawn with this site at the center, so we
  // now determine whether it's inside the Wigner-Seitz cell for this
  // lattice, centered at this particular lattice site.
  double x, y, z, sx, sy, sz;
  x = r[0];
  y = r[1];
  z = r[2];
  domain->lattice->box2lattice(x, y, z);
  sx = latsites[k][0];
  sy = latsites[k][1];
  sz = latsites[k][2];
  domain->lattice->box2lattice(sx, sy, sz);
  x = x - sx;
  y = y - sy;
  z = z - sz;
  // The coordinates (x, y, z) are now in lattice units with an origin at the
  // center of this Wigner-Seitz cell.

  // The atom is inside the Wigner-Seitz cell if
  // normal[j] . (x,y,z) / ||normal[j]||**2 <= 1 for all j.

  // That is, find whether the projection of (x,y,z) onto each normal vector
  // extends beyond the normal vector itself.
  // I use the equivalent relation here,
  // normal[j] . (x,y,z) - ||normal[j]||**2 <= 0 for all j.
  double norm2sq, diff;
  for ( int i = 0; i < nnormal; i++ ) {
    // Skip zero-length normal vectors (this should never happen)
    if ( fabs(normal[i][0]) < SMALL and fabs(normal[i][1]) < SMALL and
         fabs(normal[i][2]) < SMALL )
      continue;
    norm2sq = normal[i][0]*normal[i][0] + normal[i][1]*normal[i][1]
                  + normal[i][2]*normal[i][2];
    diff = x * normal[i][0] + y * normal[i][1] + z * normal[i][2] - norm2sq;
    if ( diff > 0.0 )
      return false;
  }

  return true;

}

/****************************************************************************/

bool ComputeFrenkel::tag_is_already_in_occupancy_list // {{{1
        (tagint tag, int site) {

  for ( int i = 0; (i < MAX_OCCUPANTS) and (occupant_tag[site][i] > 0); i++ )
    if ( occupant_tag[site][i] == tag ) return true;
  return false;
}

/****************************************************************************/

int ComputeFrenkel::next_free_occupant_tag_index (int s, int linenum) { // {{{1

  // Returns the index of the next available occupant tag for site s
  int i;
  for ( i = 0; i < MAX_OCCUPANTS and occupant_tag[s][i] > 0; i++ )
    ; // that is, just increment i until we reach the next available index
  if ( i >= MAX_OCCUPANTS )
    error->one (__FILE__, linenum, "Greater than " str(MAX_OCCUPANTS)
      " atoms near a site; incorrect lattice, perhaps?");
  return i;

}

/****************************************************************************/

void ComputeFrenkel::rescale_lattice_sites () {

  if ( not rescale ) return;

  double xprd0, yprd0, zprd0;
  xprd0 = (old_boxhi[0] - old_boxlo[0]);
  yprd0 = (old_boxhi[1] - old_boxlo[1]);
  zprd0 = (old_boxhi[2] - old_boxlo[2]);

  double xcontract, ycontract, zcontract;
  xcontract = domain->xprd / xprd0;
  ycontract = domain->yprd / yprd0;
  zcontract = domain->zprd / zprd0;

  for ( int k = 0; k < nlatsites; k++ ) {
    latsites[k][0] = (latsites0[k][0] - old_boxlo[0]) * xcontract
      + domain->boxlo[0];
    latsites[k][1] = (latsites0[k][1] - old_boxlo[1]) * ycontract
      + domain->boxlo[1];
    latsites[k][2] = (latsites0[k][2] - old_boxlo[2]) * zcontract
      + domain->boxlo[2];
  }

}

/****************************************************************************/

template <typename TYPE> void ComputeFrenkel::reallocate_array // {{{1
    (TYPE** &array, int x1, int y1, int x2, int y2) {

  // Reallocates array[x1][y1] to array[x2][y2].  Both x2 and y2 can differ
  // from x1 and y1.  Note that if y1 == y2, you should use memory->grow.
  TYPE** arr2 = array; // Now we just need a dee2
  memory->create (array, x2, y2, "ComputeFrenkel:reallocate2");
  for ( int i = 0; i < x2; i++ )
    for ( int j = 0; j < y2; j++ )
      if ( i < x1 and j < y1 )
        array[i][j] = arr2[i][j];
      else
        array[i][j] = TYPE();
  memory->destroy (arr2);

}

/****************************************************************************/

template <typename TYPE> void ComputeFrenkel::reallocate_array // {{{1
    (TYPE*** &array, int x1, int y1, int z1, int x2, int y2, int z2) {

  // Reallocates array[x1][y1][z1] to array[x2][y2][z2].  x2, y2, and z2 can
  // differ from x1, y1, and z1.  Note that if y1 == y2 and z1 == z2, you
  // should use memory->grow.
  TYPE*** arr2 = array; // Now we just need a dee2
  memory->create (array, x2, y2, z2, "ComputeFrenkel:reallocate3");
  for ( int i = 0; i < x2; i++ )
    for ( int j = 0; j < y2; j++ )
      for ( int k = 0; k < z2; k++ )
        if ( i < x1 and j < y1 and k < z1 )
          array[i][j][k] = arr2[i][j][k];
        else
          array[i][j][k] = TYPE();
  memory->destroy (arr2);

}

/****************************************************************************/

int ComputeFrenkel::process_neighbor (int x, int y, int z) { // {{{1

  // Returns the rank of the process with relative coordinates (x,y,z).
  // This process is (0,0,0).
  int rank = MPI_PROC_NULL;
  if ( x == 0 and y == 0 and z == 0 )
    return comm->me;

  int a = comm->myloc[0], b = comm->myloc[1], c = comm->myloc[2];
  if ( domain->xperiodic )
    a = (a + comm->procgrid[0] + x) % comm->procgrid[0];
  if ( domain->yperiodic )
    b = (b + comm->procgrid[1] + y) % comm->procgrid[1];
  if ( domain->zperiodic )
    c = (c + comm->procgrid[2] + z) % comm->procgrid[2];
  if ( a < 0 or b < 0 or c < 0 ) {
    rank = MPI_PROC_NULL;
    error->warning (FLERR,
          "Domain is inconsistent (got MPI_PROC_NULL next door)");
  }
  else
    rank = comm->grid2proc[a][b][c];
  return rank;
}

/****************************************************************************/

void ComputeFrenkel::turnoffoutput () { // {{{1
  // Turn off output and logging
  old_screen = lmp->screen;
  old_logfile = lmp->logfile;
  lmp->screen = nullptr;
  lmp->logfile = nullptr;
}

/****************************************************************************/

void ComputeFrenkel::revertoutput () { // {{{1
  // Revert output and logging to their previous values
  lmp->screen = old_screen;
  lmp->logfile = old_logfile;
}

/****************************************************************************/

double ComputeFrenkel::memory_usage () { // {{{1

  double nbytes = 0.0;
  if ( nlatsites == 0 ) return nbytes;
  nbytes += (atom->nmax) * sizeof(mindist[0]);
  nbytes += nlatsites * sizeof(site_mindist[0]);
  nbytes += (nlatsites + nlatghosts) * sizeof(noccupants[0]);
  nbytes += nlatsites * MAX_OCCUPANTS * sizeof(occupant_tag[0]);
  nbytes += nnormal * 3 * sizeof(normal[0][0]);
  nbytes += (nlatsites + nlatghosts) * 3 * sizeof(latsites[0][0]);
  nbytes += (nlatsites + nlatghosts) * sizeof(site_tag[0]);
  nbytes += nlatbins[0] * nlatbins[1] * nlatbins[2] * nlatbins[3]
    * sizeof(latbins[0][0][0][0]);
  nbytes += (nlatsites + nlatghosts) * nlist[0].size()
    * sizeof(tagint); // nlist
  nbytes += nlatsites * sizeof(tagint); // clusterID
  nbytes += 2 * noccupied * sizeof(int); // cluster_size, cluster_nsites
  nbytes += 3 * noccupied * sizeof(double); // cluster_center
  nbytes += noccupied * sizeof(tagint); // occupied_cluster_ID;
  return nbytes;

}

// vim: foldmethod=marker ts=2 sts=2 sw=2
