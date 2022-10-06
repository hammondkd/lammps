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

   Updated July 8, 2022 by the author.
------------------------------------------------------------------------- */

// NOTE:  This compute style requires modify_params to be virtual in compute.h

#ifdef COMPUTE_CLASS

ComputeStyle(frenkel,ComputeFrenkel)

#else

#ifndef LMP_COMPUTE_FRENKEL_H
#define LMP_COMPUTE_FRENKEL_H

#include "compute.h"
#include "memory.h"
#include <list>
#include "region.h"

namespace LAMMPS_NS {

class ComputeFrenkel : public Compute {
  public :
    ComputeFrenkel (class LAMMPS*, int, char**);
    ~ComputeFrenkel ();
    void modify_params (int, char**);

    void init ();
    int pack_reverse_comm (int, int, double *);
    void unpack_reverse_comm (int, int *, double *);

    void compute_vector ();
    void compute_array ();
    void compute_peratom ();
    void compute_local ();
    double memory_usage ();

  private :
    friend class DumpFrenkel;

    //int iregion;
    //char* idregion;
    Region* region;
    char* sitefile;
    int ifgroup, fgroupbit;
    bool rescale;

    double* mindist;
    double* site_mindist;
    int* noccupants;
    tagint** occupant_tag;
    int nnormal;
    double** normal;
    double cut_vac, cut_int, cutoff, binwidth;
    int nlatsites, nlatghosts;
    double** latsites;
    double** latsites0;
    tagint* site_tag;
    tagint first_local_tag;
    int nlatbins[4];
    int**** latbins;
    class std::list<tagint> *nlist;  // Neighbor list for SITES
    tagint* clusterID;       // Per-site vector
    int* cluster_size;       // Negative => vacancy; positive => interstitial
    int* cluster_nsites;     // Number of sites involved in cluster
    double** cluster_center; // Geometric center of cluster in x,y,z
    int noccupied;
    tagint* occupied_cluster_ID;  // Per-cluster vector, length noccupied
    double old_boxlo[3], old_boxhi[3];

    bigint invoked_find_defects;
    bigint invoked_find_clusters;
    bigint invoked_construct_WS_cell;

    FILE *old_screen, *old_logfile;

    void create_lattice_sites ();
    void put_sites_in_bins ();
    int site_tag2index (tagint);
    void exchange_lattice_ghosts ();
    void construct_WS_cell ();
    bool inside_WS_cell (int, int);
    void find_defects ();
    void find_clusters ();
    void find_occupied_clusters ();
    int clusterID2occupied_index (int);
    void find_closest_bin (double*, int&, int&, int&);
    void bin_pbc (int&, int&, int&);
    bool tag_is_already_in_occupancy_list (tagint, int);
    int next_free_occupant_tag_index (int, int);
    template <typename TYPE> void reallocate_array (TYPE**&, int, int, int,
      int);
    template <typename TYPE> void reallocate_array (TYPE***&, int, int, int,
      int, int, int);
    void construct_site_nlists ();
    void rescale_lattice_sites ();

    int process_neighbor (int, int, int);
    void turnoffoutput ();
    void revertoutput ();

    static int compareIDs (const void*, const void*);

    template <typename TYPE> TYPE ****grow (TYPE ****&array,
        int n1, int n2, int n3, int n4, const char *name) {

      if ( array == NULL ) return memory->create(array,n1,n2,n3,n4,name);

      bigint nbytes = static_cast<bigint>(sizeof(TYPE)) * n1*n2*n3*n4;
      TYPE *data = static_cast<TYPE*> (memory->srealloc (array[0][0][0], nbytes, name));
      nbytes = static_cast<bigint> (sizeof(TYPE*)) * n1*n2*n3;
      TYPE **cube = static_cast<TYPE**> (memory->srealloc (array[0][0], nbytes, name));
      nbytes = static_cast<bigint> (sizeof(TYPE**)) * n1*n2;
      TYPE ***plane = static_cast<TYPE***> (memory->srealloc (array[0], nbytes, name));
      nbytes = static_cast<bigint> (sizeof(TYPE***)) * n1;
      array = static_cast<TYPE****> (memory->srealloc (array, nbytes, name));

      int i, j, k;
      bigint m1, m2, m3;
      bigint n = 0;
      for ( i = 0; i < n1; i++ ) {
        m2 = static_cast<bigint>(i) * n2;
        array[i] = &plane[m2];
        for ( j = 0; j < n2; j++ ) {
          m1 = static_cast<bigint>(i) * n2 + j;
          m2 = static_cast<bigint>(i) * n2 * n3 + j * n3;
          plane[m1] = &cube[m2];
          for ( k = 0; k < n3; k++ ) {
            m1 = static_cast<bigint>(i) * n2 * n3 + j * n3 + k;
            cube[m1] = &data[n];
            n += n4;
          }
        }
      }

      return array;

    }

}; // end class ComputeFrenkel

} // end namespace

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal compute frenkel command

Usually self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

Possible causes specific to this class:
 - Providing something other than 3 arguments
 - Providing no arguments to compute_modify
 - Providing too many arguments to compute_modify

E: Cannot use compute style frenkel unless atoms have IDs

All atoms must have ID's for this to work.

E: Use of compute style frenkel with undefined lattice

The lattice points are used as the "normal" locations of the atoms.  If no
lattice is defined, the assumed-site model is not defined either.

E: Illegal compute_modify command: ... values must be positive.

Self-explanatory.

E: Compute_modify (group|region) does not exist

Self-explanatory.

E: Unable to find lattice sites group for dump style frenkel

This should basically never happen; please report a bug.

E: Greater than ... atoms near a site

This indicates that your system has too many atoms near a single site.  Make
certain you are using the lattice that corresponds to your atomic
configuration.  If you are 100% certain that there should be more than eight
atoms per site (i.e., eight atoms at one lattice site), then you can recompile
with MAX_OCCUPANTS at a higher value.  This might happen, for example, if you
have a small atom like helium bouncing around a much larger atom.

W: Domain is inconsistent (got MPI_PROC_NULL next door)

This indicates something is very wrong.  Strongly consider reporting a bug.

W: Compute style frenkel does not have degrees of freedom

Self-explanatory.

W: Compute style frenkel does not use the dynamic key word

Self-explanatory.

W: Compute style frenkel does not use the thermo key word

Self-explanatory.  Note that this is the "thermo" key word to compute_modify,
NOT the thermo /command/.

W: Did not find cluster index

This should pretty much never happen; it indicates that the compute just tried
to look up a cluster ID that doesn't even exist.  Seeing this would indicate
a bug.

*/
// vim: foldmethod=marker ts=2 sts=2 sw=2
