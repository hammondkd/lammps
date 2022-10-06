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

#ifdef DUMP_CLASS

DumpStyle(frenkel,DumpFrenkel)

#else

#ifndef LMP_DUMP_FRENKEL_H

#define LMP_DUMP_FRENKEL_H

#include "dump.h"
#include "compute_frenkel.h"

namespace LAMMPS_NS {

class DumpFrenkel : public Dump {
   public :
      DumpFrenkel (class LAMMPS*, int, char** );
      ~DumpFrenkel();
      double memory_usage ();
   private :
      char *compute_name;
      int compute_id;
      bool compute_created_for_me;
      bool scale_flag;
      bool compute_has_been_modified;
      int* offsite;
      int* vacant;

      bool use_WS_cell;
      char *columns;             // column labels
      double dr;

      int modify_param (int, char**);
      int count (void);
      void write_data (int, double*);
      void init_style (void);
      void write_header (bigint);
      void pack (int*);
      //int pack_reverse_comm (int, int, double*);
      //void unpack_reverse_comm (int, int*, double*);

   // Header choices
      typedef void (DumpFrenkel::*FnPtrHeader)(bigint);
      FnPtrHeader header_choice;          // pointer to header functions
      void header_binary (bigint);
      void header_binary_triclinic (bigint);
      void header_item (bigint);
      void header_item_triclinic (bigint);

   // Private, non-standard routines
      void create_default_compute (void);

}; // End of class

} // End of namespace

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot use dump style frenkel unless atoms have IDs

All atoms must have ID's for this to work.

E: Dump style frenkel requires an atom map

You must define an atom map, such as with "atom_modify map array"; the atoms
are dumped by reverse-lookup for the "cell" method.

E: Use of dump style frenkel with undefined lattice

The last defined lattice command is used to generate the lattice sites against
which the atoms are compared.  It must exist.

E: Dump style frenkel does not yet support triclinic lattices

Self-explanatory.

E: Compute ID does not exist

Self-explanatory.

E: Compute paired with dump style frenkel must be of compute style frenkel

Self-explanatory.

E: Bad dr value in dump_modify command

If dr <= 0, that means you're asking to mark sites as empty if something is
a /negative/ distance away.  Riiight.

W: Previous modifications to Frenkel compute may have been destroyed

If you modify a compute, then modify it again indirectly (say, by another
dump that is also linked to it), you might overwrite some of those changes
if you're not careful.

W: Something is wrong; you just overwrote a Frenkel compute with the default

This should probably never happen.  If it does, consider reporting a bug.

E: Invalid compute_modify command.

This happens because this dump style might actually modify the underlying
compute; make sure your syntax matches what it should look like.  In such
cases, the compute_modify command will look the same as the dump_modify
command, at least for the key words interpreted by both commands.

*/

// vim: foldmethod=marker tabstop=3 expandtab
