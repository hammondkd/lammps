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

#include "dump_frenkel.h"
#include <mpi.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "lattice.h"
#include "modify.h"
#include "region.h"
#include "update.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

// Parameters local to this file
#define DumpFrenkel_vactype 0

using namespace LAMMPS_NS;
using namespace std;

DumpFrenkel::DumpFrenkel (LAMMPS *lmp, int narg, char **arg) : // {{{1
      Dump (lmp, narg, arg) {

  if ( narg != 5 )
    error->all (FLERR,"Illegal dump frenkel command");

  // Error if no atom tags are defined or there is no atom map
  if ( not atom->tag_enable )
    error->all (FLERR, "Cannot use dump style frenkel unless atoms have IDs");

  if ( not domain->lattice )
    error->all (FLERR,"Use of dump style frenkel with undefined lattice");

  // Default dr is 0.3 lattice units; this will be the minimum such distance
  // for rotated or non-cubic lattices, and will likely be incorrect for
  // anything other than unrotated primitive cubic lattices (SC, BCC, FCC).
  // It it STRONGLY recommended to change the dr values for rotated or
  // non-cubic lattices using the dump_modify command.
  double a_min = MIN( MIN( domain->lattice->xlattice,
      domain->lattice->ylattice), domain->lattice->zlattice);
  double a_max = MAX( MAX( domain->lattice->xlattice,
      domain->lattice->ylattice), domain->lattice->zlattice);
  dr = 0.3 * a_min;

  // Settings and defaults
//  comm_reverse = 0;
  scale_flag = true;
  columns = NULL;
  compute_created_for_me = false;
  compute_name = NULL;
  compute_id = -1;
  compute_has_been_modified = false;
  offsite = vacant = NULL;
  use_WS_cell = false; // Distance method is the default for dump_frenkel

}

/****************************************************************************/

DumpFrenkel::~DumpFrenkel () { // {{{1

  // I WILL DESTROY YOU!
  // (Sorry to the non-Futurama fans out there who don't get this joke)

  delete [] columns;
  // If the user gave us a compute to be used for other purposes, then leave
  // it alone.  If I created it, destroy it now.
  if ( compute_name != NULL and not compute_created_for_me ) {
    int id = modify->find_compute(compute_name);
    if ( id > 0 )
      modify->delete_compute(compute_name);
  }
  delete [] compute_name;

}

/****************************************************************************/

void DumpFrenkel::init_style () { // {{{1

  // if user asked for a dump using method "cell" and there's no atom map,
  // we are going to be in trouble
  if ( use_WS_cell and not atom->map_style )
    error->all (FLERR, "Dump style frenkel with 'cell' method requires an"
      " atom map");

  // size_one is part of dump.h
  size_one = 5;

  // Set default format
  delete [] format;
  if ( format_line_user ) {
    int n = strlen(format_line_user) + 2;
    format = new char[n];
    strcpy(format,format_line_user);
    strcat(format,"\n");
  }
  else {
    char *str;
    str = (char*) TAGINT_FORMAT " %d %g %g %g";
    int n = strlen(str) + 2;
    format = new char[n];
    strcpy (format, str);
    strcat (format, "\n");
  }

  // Set up boundary string (e.g., pp pp fm)
  domain->boundary_string(boundstr);

  // Set up column string
  delete [] columns;
  if (scale_flag == 0) {
    columns = new char[14];
    strcpy(columns, "id type x y z");
  }
  else {
    columns = new char[17];
    strcpy(columns, "id type xs ys zs");
  }

  // Open single file, one time only
  if ( multifile == 0 ) openfile();

  // Set up header choices
  if ( binary and domain->triclinic == 0 )
    header_choice = &DumpFrenkel::header_binary;
  else if ( binary and domain->triclinic == 1 )
  //  header_choice = &DumpFrenkel::header_binary_triclinic;
    error->all (FLERR,
      "Dump style frenkel does not yet support triclinic lattices"); // FIXME
  else if ( !binary and domain->triclinic == 0 )
    header_choice = &DumpFrenkel::header_item;
  else if ( !binary and domain->triclinic == 1 )
    //header_choice = &DumpFrenkel::header_item_triclinic;
    error->all(FLERR,
      "Dump style frenkel does not yet support triclinic lattices"); // FIXME

  // If compute has not been created, create a new compute and initialize it
  if ( compute_id < 0 ) {
    create_default_compute();
    // Computes get initialized before dumps do
    modify->compute[compute_id]->init();
  }

}

/****************************************************************************/

int DumpFrenkel::modify_param (int narg, char **arg) { // {{{1

  // Use pre-existing compute (this is the way to go if you want both dump
  // files AND clustering data)
  if ( strcmp(arg[0],"compute") == 0 ) {
    if ( narg < 2 ) error->all(FLERR, "Illegal dump_modify command");
    compute_created_for_me = true;
    delete [] compute_name;
    compute_name = new char[strlen(arg[1])+1];
    strcpy(compute_name, arg[1]);
    compute_id = modify->find_compute(compute_name);
    if ( compute_id < 0 )
      error->all (FLERR, "Compute ID does not exist");
    // Make sure this is actually a Frenkel style compute!
    if ( strcmp (modify->compute[compute_id]->style, "frenkel") != 0 )
      error->all (FLERR, "Compute paired with dump style frenkel must be of compute style frenkel");
    if ( compute_has_been_modified and comm->me == 0 )
      error->warning (FLERR,
        "Previous modifications to Frenkel compute may have been destroyed");
    compute_has_been_modified = false;
    return 2;
  }
  // Scaled vs. unscaled
  if ( strcmp(arg[0],"scale") == 0 ) {
    if ( narg < 2 ) error->all(FLERR, "Illegal dump_modify command");
    if ( strcmp(arg[1],"yes") == 0 )
       scale_flag = true;
    else if ( strcmp(arg[1],"no") == 0 )
       scale_flag = false;
    else
       error->all(FLERR, "Illegal dump_modify command");
    return 2;
  }
  // New compute region
  else if ( strcmp(arg[0],"region") == 0 ) {
    if ( narg < 2 ) error->all(FLERR, "Illegal dump_modify command");
    if ( compute_id < 1 ) // Compute does not exist yet; create it
      create_default_compute();
    compute_has_been_modified = true;
    // Now modify the compute
    modify->compute[compute_id]->modify_params(2, arg);
    return 2;
  }
  // New value of dr
  else if ( strcmp(arg[0],"dr") == 0 ) {
    if ( narg < 2 ) error->all(FLERR, "Illegal dump_modify command");
    dr = atof (arg[1]);
    if ( dr <= 0.0 )
       error->all (FLERR, "Bad dr value in dump_modify command");
    return 2;
  }
  // New Frenkel group
  else if ( strcmp(arg[0],"frenkelgroup") == 0 ) {
    if ( narg < 2 ) error->all (FLERR, "Illegal dump_modify command");
    if ( compute_id < 1 ) // Compute does not exist yet; create it
      create_default_compute();
    compute_has_been_modified = true;
    modify->compute[compute_id]->modify_params(2, arg);
    return 2;
  }
  // Change defect identification method
  else if ( strcmp(arg[0],"method") == 0 ) {
    if ( narg < 2 ) error->all (FLERR, "Illegal dump_modify command");
    if ( strcmp (arg[1],"cell") == 0 )
      use_WS_cell = true;
    else if ( strcmp (arg[1],"distance") == 0 )
      use_WS_cell = false;
    else
      error->all (FLERR, "Illegal dump_modify command");
    return 2;
  }
  // Rescale?
  else if ( strcmp(arg[0],"rescale") == 0 ) {
    if ( narg < 2 ) error->all (FLERR, "Illegal dump_modify command");
    if ( compute_id < 1 ) // Compute does not exist yet; create it
      create_default_compute();
    compute_has_been_modified = true;
    modify->compute[compute_id]->modify_params(2, arg);
    return 2;
  }
  // Get sites from file
  else if ( strcmp (arg[0],"site_file") == 0 ) {
    if ( narg < 2 ) error->all (FLERR, "Illegal dump_modify command");
    if ( compute_id < 1 ) // Compute does not exist yet; create it
      create_default_compute();
    compute_has_been_modified = true;
    modify->compute[compute_id]->modify_params(2, arg);
    return 2;
  }
  // If we didn't find something we understood, tell the Dump class "FAIL."
  return 0;

}

/****************************************************************************/

void DumpFrenkel::create_default_compute () { // {{{1

  if ( compute_has_been_modified ) // should NEVER happen
    error->warning (FLERR, "Something is wrong;"
      " you just overwrote a Frenkel compute with the default");
  compute_has_been_modified = false;
  compute_created_for_me = false;
  delete [] compute_name;
  compute_name = new char[strlen("compute_")+strlen(id)+1];
  strcpy (compute_name, "compute_");
  strcat (compute_name, id);
  compute_id = modify->find_compute(compute_name);
  if ( compute_id < 0 ) { // Does not exist; create it
    char **arg = new char*[3];
    arg[0] = compute_name;
    arg[1] = group->names[igroup];
    arg[2] = new char[strlen("frenkel")+1];
    strcpy(arg[2],"frenkel");
    modify->add_compute (3, arg);
    compute_id = modify->find_compute(compute_name);
    class ComputeFrenkel *compute = NULL;
    compute = static_cast<class ComputeFrenkel*> (modify->compute[compute_id]);
    use_WS_cell = false;
    delete [] arg[2];
    delete [] arg;
  }
  // No else:  we must have already created this for this dump id, so don't do
  // it again (and again, and again...)

}

/****************************************************************************/

void DumpFrenkel::write_header (bigint ndump) { // {{{1

  if ( multiproc )
    (this->*header_choice)(ndump);
  else if ( me == 0 )
    (this->*header_choice)(ndump);

}

/****************************************************************************/

// Header routines {{{1
void DumpFrenkel::header_item (bigint ndump) { // {{{2

  fprintf(fp,"ITEM: TIMESTEP\n");
  fprintf(fp,BIGINT_FORMAT "\n",update->ntimestep);
  fprintf(fp,"ITEM: NUMBER OF ATOMS\n");
  fprintf(fp,BIGINT_FORMAT "\n",ndump);
  fprintf(fp,"ITEM: BOX BOUNDS %s\n",boundstr);
  fprintf(fp,"%g %g\n",boxxlo,boxxhi);
  fprintf(fp,"%g %g\n",boxylo,boxyhi);
  fprintf(fp,"%g %g\n",boxzlo,boxzhi);
  fprintf(fp,"ITEM: ATOMS %s\n",columns);

}

/****************************************************************************/

void DumpFrenkel::header_binary (bigint ndump) { // {{{2

  fwrite(&update->ntimestep, sizeof(bigint), 1, fp);
  fwrite(&ndump,sizeof(bigint),1,fp);
  fwrite(&domain->triclinic,sizeof(int),1,fp);
  fwrite(&domain->boundary[0][0], 6*sizeof(int), 1, fp);
  fwrite(&boxxlo, sizeof(double), 1, fp);
  fwrite(&boxxhi, sizeof(double), 1, fp);
  fwrite(&boxylo, sizeof(double), 1, fp);
  fwrite(&boxyhi, sizeof(double), 1, fp);
  fwrite(&boxzlo, sizeof(double), 1, fp);
  fwrite(&boxzhi, sizeof(double), 1, fp);
  fwrite(&size_one, sizeof(int), 1, fp);
  if (multiproc) {
     int one = 1;
     fwrite(&one,sizeof(int),1,fp);
  }
  else
     fwrite(&nprocs,sizeof(int),1,fp);

}

/****************************************************************************/

int DumpFrenkel::count() { // {{{1

  class ComputeFrenkel *compute = NULL;
  compute = static_cast<class ComputeFrenkel*> (modify->compute[compute_id]);
  if ( compute->invoked_find_defects != update->ntimestep )
    compute->find_defects();

  vacant = new int[compute->nlatsites];
  offsite = new int[atom->nlocal+atom->nghost];

  for ( int i = 0; i < atom->nlocal + atom->nghost; i++ )
    offsite[i] = false;
  for ( int k = 0; k < compute->nlatsites; k++ )
    vacant[k] = false;

  // Wigner-Seitz method:  dump contents of empty or 2+ occupied cells
  if ( use_WS_cell ) {
    for ( int k = 0; k < compute->nlatsites; k++ ) {
      if ( compute->noccupants[k] == 0 )
        vacant[k] = true;
      else if ( compute->noccupants[k] >= 2 ) {
        vacant[k] = true;
        for ( int i = 0; i < compute->noccupants[k]; i++ ) {
          tagint id = compute->occupant_tag[k][i];
          int index = atom->map(id);
          if ( index >= 0 )
            offsite[index] = true;
        }
      }
    }
  }
  else { // Distance method: dump anything off-lattice and all vacant sites
    for ( int k = 0; k < compute->nlatsites; k++ )
      if ( compute->site_mindist[k] > dr )
        vacant[k] = true;
    for ( int i = 0; i < atom->nlocal; i++ )
      if ( compute->mindist[i] > dr )
        offsite[i] = true;
  }

  // Flag and print heteroatoms and out-of-region atoms
  Region* region = compute->region;
  double** x = atom->x;
  for ( int i = 0; i < atom->nlocal; i++ ) {
    if ( not (atom->mask[i] & groupbit) )
      offsite[i] = false;
    else if ( (atom->mask[i] & groupbit) and
        not (atom->mask[i] & compute->fgroupbit) )
      offsite[i] = true; // always print atoms NOT in the frenkel group
/*    else if ( (atom->mask[i] & groupbit) and
        compute->mindist[i] > dr and
          //not ( domain->regions[iregion]->match (x[i][0], x[i][1], x[i][2]) )
          not ( region->match (x[i][0], x[i][1], x[i][2]) )
        )
      offsite[i] = true; */
  }

  // Now actually do the counting!
  int noffsite = 0, nempty = 0;
  for ( int k = 0; k < compute->nlatsites; k++ )
    if ( vacant[k] )
      nempty += 1;
  for ( int n = 0; n < atom->nlocal; n++ )
    if ( offsite[n] )
      noffsite += 1;
  return noffsite + nempty;

}

/****************************************************************************/

void DumpFrenkel::pack (int *ids) { // {{{1

  double invxprd = 1.0 / domain->xprd;
  double invyprd = 1.0 / domain->yprd;
  double invzprd = 1.0 / domain->zprd;
  double lamda[3];
  int m, n, p;
  m = n = p = 0;

  class ComputeFrenkel* compute = NULL;
  compute = static_cast<class ComputeFrenkel*> (modify->compute[compute_id]);

  // Flag and print off-lattice atoms
  for ( int i = 0; i < atom->nlocal; i++ ) {
    if ( offsite[i] ) {
      buf[m++] = atom->tag[i];
      buf[m++] = atom->type[i];
      if ( domain->triclinic ) {
        domain->x2lamda (atom->x[i], lamda);
        buf[m++] = lamda[0];
        buf[m++] = lamda[1];
        buf[m++] = lamda[2];
      }
      else if ( scale_flag ) {
        // As in pack_scale_noimage from the DumpAtom class
        buf[m++] = (atom->x[i][0] - boxxlo) * invxprd;
        buf[m++] = (atom->x[i][1] - boxylo) * invyprd;
        buf[m++] = (atom->x[i][2] - boxzlo) * invzprd;
      }
      else {
        // As in pack_noscale_noimage from the DumpAtom class
        buf[m++] = atom->x[i][0];
        buf[m++] = atom->x[i][1];
        buf[m++] = atom->x[i][2];
      }
    }
  }

  // Now add the vacancies
  for ( int k = 0; k < compute->nlatsites; k++ ) {
    if ( vacant[k] ) {
      buf[m++] = compute->site_tag[k];
      buf[m++] = DumpFrenkel_vactype;
      if ( domain->triclinic ) {
        domain->x2lamda(compute->latsites[k], lamda);
        buf[m++] = lamda[0];
        buf[m++] = lamda[1];
        buf[m++] = lamda[2];
      }
      else if ( scale_flag ) {
        buf[m++] = (compute->latsites[k][0] - boxxlo) * invxprd;
        buf[m++] = (compute->latsites[k][1] - boxylo) * invyprd;
        buf[m++] = (compute->latsites[k][2] - boxzlo) * invzprd;
      }
      else {
        buf[m++] = compute->latsites[k][0];
        buf[m++] = compute->latsites[k][1];
        buf[m++] = compute->latsites[k][2];
      }
    }
  }

  delete [] offsite;
  delete [] vacant;

}

/****************************************************************************/

void DumpFrenkel::write_data (int n, double* mybuf) { // {{{1

  if ( binary ) { // Binary file
    n *= size_one;
    fwrite (&n, sizeof(int), 1, fp);
    fwrite (mybuf, sizeof(double), n, fp);
  }
  else { // ASCII file
    int m = 0;
    for ( int i = 0; i < n; i++ ) {
      fprintf (fp, format, static_cast<tagint>(mybuf[m]),
        static_cast<int>(mybuf[m+1]), mybuf[m+2], mybuf[m+3], mybuf[m+4]);
      m += size_one;
    }
  }

}

/****************************************************************************/
/*
int DumpFrenkel::pack_reverse_comm (int n, int first, double* buf) { // {{{1

  int m, last;
  m = 0;
  last = first + n;
  for ( int i = first; i < last; i++ )
    buf[m++] = static_cast<double>(offsite[i]);
  return m;

}
*/
/****************************************************************************/
/*
void DumpFrenkel::unpack_reverse_comm (int n, int* list, double* buf) { // {{{1

  int j = 0, m = 0;
  for ( int i = 0; i < n; i++ ) {
    j = list[i];
    offsite[j] = ( offsite[j] or static_cast<int>(buf[m++]) );
  }

}
*/
/****************************************************************************/

double DumpFrenkel::memory_usage () { // {{{1

  double bytes = 0.0;
  if ( columns ) bytes += (strlen(columns)+1) * sizeof(char);
  if ( compute_name ) bytes += (strlen(compute_name)+1) * sizeof(char);
  bytes += ( atom->nlocal + atom->nghost ) * sizeof(int);
  class ComputeFrenkel* compute = NULL;
  compute = static_cast<class ComputeFrenkel*> (modify->compute[compute_id]);
  if ( compute ) bytes += compute->nlatsites * sizeof(int);

  return bytes;
}

// }}}

/****************************************************************************/

// vim: foldmethod=marker tabstop=2 expandtab
