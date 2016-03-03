#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/lac/precondition.h>

#include <Epetra_Map.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace dealii;

class ParameterReader : public Subscriptor
{
  public:
    ParameterReader(ParameterHandler &);
    void read_parameters(const std::string);

  private:
    void declare_parameters();
    ParameterHandler &prm;
};

ParameterReader::ParameterReader(ParameterHandler &paramhandler)
 :
  prm(paramhandler)
{}

void ParameterReader::declare_parameters()
{
  prm.enter_subsection ("Mesh Information");
    prm.declare_entry ("Gmesh Input" , "false", Patterns::Bool());
    prm.declare_entry ("Input File Name", "",Patterns::Anything());
    prm.declare_entry ("Initial Level" , "2", Patterns::Integer(0,255));
    prm.declare_entry ("Max Level" , "0", Patterns::Integer(0,255));
    prm.declare_entry ("Min Level" , "0", Patterns::Integer(0,255));
    prm.declare_entry ("X-axis min" , "-0.5" , Patterns::Double(-10,10));
    prm.declare_entry ("X-axis max" , "+0.5" , Patterns::Double(-10,10));
    prm.declare_entry ("Y-axis min" , "-0.5" , Patterns::Double(-10,10));
    prm.declare_entry ("Y-axis max" , "+0.5" , Patterns::Double(-10,10));
    prm.declare_entry ("Z-axis min" , "-0.5" , Patterns::Double(-10,10));
    prm.declare_entry ("Z-axis max" , "+0.5" , Patterns::Double(-10,10));
    prm.declare_entry ("Small Radius" , "+0.5" , Patterns::Double(-1,1));
    prm.declare_entry ("Big Radius" , "+0.5" , Patterns::Double(-1,1));
    prm.declare_entry ("Big Aspect Ratio" , "+0.0" , Patterns::Double(-1,1));
  prm.leave_subsection ();

  prm.enter_subsection ("Particle");
    prm.declare_entry ("Immobile Particle" , "false" , Patterns::Bool());
    prm.declare_entry ("Factor of Safe Guard", "1" , Patterns::Double(0,10000));
    prm.declare_entry ("No. Particle", "0" , Patterns::Integer(0,10000));
    prm.declare_entry ("Random Particles" , "false" , Patterns::Bool());
    prm.declare_entry ("Particle Radius" , "0.15" , Patterns::Double(0,1000));
    prm.declare_entry ("Viscosity Factor" , "100.0" , Patterns::Double(1,100000));
    prm.declare_entry ("Aspect Ratio" , "1.0" , Patterns::Double(-1000,1000));
    prm.declare_entry ("Aspect ratio for Cylinder" , "0.0" , Patterns::Double(-1000,1000));
    prm.declare_entry ("Orientation for X-axis" , "0.0" , Patterns::Double(-1,1));
    prm.declare_entry ("Orientation for Y-axis" , "0.0" , Patterns::Double(-1,1));
    prm.declare_entry ("Orientation for Z-axis" , "0.0" , Patterns::Double(-1,1));
    prm.declare_entry ("Viscosity Method" , "0" , Patterns::Integer(0,10));
  prm.leave_subsection ();

  prm.enter_subsection ("Equation");
    prm.declare_entry ("Time Interval" , "0.01" , Patterns::Double(0.0001,1));
    prm.declare_entry ("No. of Time Step" , "1" , Patterns::Integer(0,100000));
    prm.declare_entry ("Total Viscosity" , "1.0" , Patterns::Double(0.0 , 100000));
    prm.declare_entry ("Relaxation Time" , "0.0" , Patterns::Double(0,1000));
    prm.declare_entry ("Beta" , "0.5" , Patterns::Double(0,1));
    prm.declare_entry ("Shear Rate" , "1.0" , Patterns::Double(0,1000));
    prm.declare_entry ("Amplitude" , "0.0" , Patterns::Double(0,1000));
    prm.declare_entry ("Angular Frequency" , "0.0" , Patterns::Double(0,1000));
  prm.leave_subsection ();

  prm.enter_subsection ("Restart");
    prm.declare_entry ("Restart" , "false" , Patterns::Bool());
    prm.declare_entry ("Check Point" , "0" , Patterns::Integer(0,1000000));
    prm.declare_entry ("Index for VTU" , "0" , Patterns::Integer(0,1000000));
  prm.leave_subsection ();
    
  prm.enter_subsection ("Problem");
    prm.declare_entry ("Dimension" , "2" , Patterns::Integer(0,10));
    prm.declare_entry ("Initial Mesh Check" , "false" , Patterns::Bool());
    prm.declare_entry ("Read Data" , "false" , Patterns::Bool());
    prm.declare_entry ("Re-Run-i" , "0" , Patterns::Integer(0,100000));
    prm.declare_entry ("Recover Trg." , "false" , Patterns::Bool());
    prm.declare_entry ("Recover Orn.", "false", Patterns::Bool());
    prm.declare_entry ("Error Stokes" , "1e-06" , Patterns::Double(0,1));
    prm.declare_entry ("Error Stress" , "1e-08" , Patterns::Double(0,1));
    prm.declare_entry ("Consti. Model Type" , "0" , Patterns::Integer(0,10));
    prm.declare_entry ("Model Parameter" , "0" , Patterns::Double(0,1));
    prm.declare_entry ("Debug Mode" , "false" , Patterns::Bool());
    prm.declare_entry ("Periodic Bnd." , "false" , Patterns::Bool());
    prm.declare_entry ("Output Period" , "0" , Patterns::Integer(0,10000));
    prm.declare_entry ("Refine Period" , "-1" , Patterns::Integer(-1,10000));
  prm.leave_subsection ();
}

void ParameterReader::read_parameters(const std::string parameter_file)
{
  declare_parameters();
  prm.read_input (parameter_file);
}

template <int dim>
class Stokes_BoundaryValues : public Function<dim>
{

  public:

    Stokes_BoundaryValues (double imposed_vel_at_wall);

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;

    virtual void vector_value_list (const std::vector<Point<dim> > &p,
                                    std::vector<Vector<double> > &values) const;

    double imposed_vel_at_wall;
};

template <int dim>
Stokes_BoundaryValues<dim>::Stokes_BoundaryValues
       (double imposed_vel_at_wall) :
Function<dim>(dim+1),
imposed_vel_at_wall (imposed_vel_at_wall)
{}

template <int dim>
double Stokes_BoundaryValues<dim>::value (const Point<dim>  &p,
                                          const unsigned int component) const
{
  double rv = 0.0;

  if (component == 0) rv = imposed_vel_at_wall*p[1];

  return rv;
}

template <int dim>
void Stokes_BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                        Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values(c) = Stokes_BoundaryValues<dim>::value (p, c);
}

template <int dim>
void Stokes_BoundaryValues<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                    std::vector<Vector<double> >   &value_list) const
{
  for (unsigned int p=0; p<points.size(); ++p)
    Stokes_BoundaryValues<dim>::vector_value (points[p], value_list[p]);
}


template <int dim>
class Viscoelastic_BoundaryValues : public Function<dim>
{
  public:
  Viscoelastic_BoundaryValues (double dt, double shrF,
                               double VisPoly, double tauD,
                               std::vector<double> &visEls_comp);

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &value) const;

  virtual void vector_value_list (const std::vector<Point<dim> > &p,
                                  std::vector<Vector<double> > &values) const;

  double dt, shrF, VisPoly, tauD;
  std::vector<double> visEls_comp;
};

template <int dim>
Viscoelastic_BoundaryValues<dim>::Viscoelastic_BoundaryValues 
(
  double dt, double shrF, double VisPoly, double tauD,
  std::vector<double> &visEls_comp
) :
Function<dim>(3*(dim-1)),
dt(dt),
shrF(shrF),
VisPoly(VisPoly),
tauD(tauD),
visEls_comp(visEls_comp)
{}

template <int dim>
double Viscoelastic_BoundaryValues<dim>::value ( const Point<dim>  &p,
                                                 const unsigned int component) const
{
  double rv = 0.0;

//   visEls_comp[0] = t11;
//   visEls_comp[1] = t22;
//   visEls_comp[2] = t12;

  switch (component)
  {
    case 0: rv = visEls_comp[0]  + 2*dt*shrF*visEls_comp[2]- (dt/tauD)*visEls_comp[0]; break;
    case 1: rv = visEls_comp[1] - (dt/tauD)*visEls_comp[1]; break;
    case 2: rv = visEls_comp[2] - (dt/tauD)*(visEls_comp[2] - VisPoly*shrF); break;
  }

  return rv;
}

template <int dim>
void Viscoelastic_BoundaryValues<dim>::vector_value ( const Point<dim> &p,
                                                      Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values(c) = Viscoelastic_BoundaryValues<dim>::value (p, c);
}

template <int dim>
void Viscoelastic_BoundaryValues<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                          std::vector<Vector<double> >   &value_list) const
{
  for (unsigned int p=0; p<points.size(); ++p)
  Viscoelastic_BoundaryValues<dim>::vector_value (points[p], value_list[p]);
}


namespace LinearSolvers
{
  template <class PreconditionerA, class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
    public:
      BlockSchurPreconditioner (
        const TrilinosWrappers::BlockSparseMatrix  &S,
        const PreconditionerMp                     &Mppreconditioner,
        const PreconditionerA                      &Apreconditioner);

      void vmult (TrilinosWrappers::MPI::BlockVector       &dst,
                  const TrilinosWrappers::MPI::BlockVector &src) const;

    private:
      const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> stokes_matrix;
      const PreconditionerMp &mp_preconditioner;
      const PreconditionerA  &a_preconditioner;
      mutable TrilinosWrappers::MPI::Vector tmp;
  };

  template <class PreconditionerA, class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
  BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &S,
                           const PreconditionerMp                    &Mppreconditioner,
                           const PreconditionerA                     &Apreconditioner)
                  :
                  stokes_matrix     (&S),
                  mp_preconditioner (Mppreconditioner),
                  a_preconditioner  (Apreconditioner),
                  tmp               (stokes_matrix->block(1,1).row_partitioner())
  {}

  template <class PreconditionerA, class PreconditionerMp>
  void BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::vmult (
    TrilinosWrappers::MPI::BlockVector       &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
  {
    a_preconditioner.vmult (dst.block(0), src.block(0));
    stokes_matrix->block(1,0).residual(tmp, dst.block(0), src.block(1));
    tmp *= -1;
    mp_preconditioner.vmult (dst.block(1), tmp);
  }
}

template <int dim>
class ViscoElasticFlow
{
  public:
    ViscoElasticFlow (ParameterHandler &);
    ~ViscoElasticFlow ();
    void run ();

  private:
    void readat ();
    void setup_dofs (bool);

    void assemble_stokes_preconditioner ();
    void build_stokes_preconditioner ();
    void assemble_stokes_system ();

    void make_peri_cell_stress ();
    void assemble_peri_DGConvect ();
    void assemble_RHSVisEls ();
    void assemble_face_term (const FEFaceValuesBase<dim>& fe_v,
                             const FEFaceValuesBase<dim>& fe_v_neighbor,
                             FullMatrix<double> &ui_vi_matrix,
                             FullMatrix<double> &ue_vi_matrix,
                             std::vector<Vector<double> > &solu) const;

    void solve (unsigned int part , unsigned int ist);

    void refine_mesh (bool matrix_init);
    void error_indicator_for_isotropic ();
    void error_indicator_for_anisotropic ();
    
    void plotting_solution (unsigned int np) const;
    void post_processing (unsigned int i, unsigned int np,
                           std::ofstream &out, std::ofstream &out_p,
      std::ofstream &out_a, std::ofstream &out_or,
      std::ofstream &out_w);

    void compute_rheology_properties ();
    
    void particle_generation ();
    void compute_particle_dyn_properties (unsigned int, 
       std::ofstream &, 
       std::ofstream &);
    void compute_stress_system (unsigned int, 
    std::ofstream &, 
    std::ofstream &);
    void particle_move ();
    void viscosity_distribution ();
    std::pair<unsigned int, double> distant_from_particles (Point<dim> &,
             double, double); 

    void level_set_2nd_adv_step ();
    void level_set_compute_normal (unsigned int);
    void level_set_reinitial_step ();
    void solve_levelSet_function ();
    
    void
    make_flux_sparsity_pattern (
      DoFHandler<dim> &dof,
      TrilinosWrappers::BlockSparsityPattern &sparsity,
      const Table<2,DoFTools::Coupling> &int_mask,
      const Table<2,DoFTools::Coupling> &flux_mask,
      bool is_periodic,
      std::vector<typename DoFHandler<dim>::active_cell_iterator> &peri_cells,
      const unsigned int subdomain_id);

    void create_triangulation ();
    void initial_refined_coarse_mesh ();
    void recover_triangulation (std::ofstream &elem);
    void recover_orientation ();
    void read_solutions ();
    void write_solutions (unsigned int ist);

    void match_periodic_faces ();
 
    void make_periodicity_constraints (DoFHandler<dim> &,
          types::boundary_id   ,
          types::boundary_id   ,
          int                  ,
          ConstraintMatrix  &);
    
    void make_periodicity_constraints (
      const typename DoFHandler<dim>::face_iterator    &face_1,
      const typename identity<typename DoFHandler<dim>::face_iterator>::type &face_2,
      ConstraintMatrix  &constraint_matrix,
      const ComponentMask  &component_mask,
      const bool   face_orientation,
      const bool   face_flip,
      const bool   face_rotation);
    
    void set_periodicity_constraints (
      const typename DoFHandler<dim>::face_iterator &face_1,
      const typename identity<typename DoFHandler<dim>::face_iterator>::type &face_2,
      const FullMatrix<double> &transformation,
      ConstraintMatrix           &constraint_matrix,
      const ComponentMask       &component_mask,
      const bool                  face_orientation,
      const bool                   face_flip,
      const bool                   face_rotation);
    
    void save_snapshot (unsigned int np);
    void resume_snapshot ();
    
    const Epetra_Comm  &trilinos_communicator;
    ConditionalOStream  pcout;
    ParameterHandler  &prm;
    
    Triangulation<dim>  triangulation;
    
    FESystem<dim>  stokes_fe;
    FESystem<dim>  stress_fe;
    FE_Q<dim>   fe_levelset;
    
    DoFHandler<dim>  stokes_dof_handler; 
    DoFHandler<dim>  stress_dof_handler;
    DoFHandler<dim>  dof_handler_levelset;
    
    ConstraintMatrix  stokes_constraints;
    ConstraintMatrix  constraint_levelset;

    std::vector<Epetra_Map> stokes_partitioner;
    std::vector<Epetra_Map> stress_partitioner;
    
    TrilinosWrappers::BlockSparseMatrix stokes_matrix;
    TrilinosWrappers::BlockSparseMatrix stress_matrix;
    TrilinosWrappers::BlockSparseMatrix stokes_preconditioner_matrix;
    TrilinosWrappers::SparseMatrix      matrix_levelset;
    
    TrilinosWrappers::BlockVector   stokes_solution, stokes_old_solution;
    TrilinosWrappers::BlockVector   stress_solution, stress_old_solution;
    TrilinosWrappers::Vector  levelset_solution, old_levelset_solution;
    TrilinosWrappers::Vector  level_set_normal_x, level_set_normal_y, level_set_normal_z;
    
    TrilinosWrappers::MPI::BlockVector  stokes_rhs, stress_rhs, rhs_levelset;
    
    TrilinosWrappers::Vector  particle_pos, viscosity_solution;

    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG> Amg_preconditioner;
    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionILU> Mp_preconditioner;
    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG>  T_preconditioner;

    //Basic
    const MappingQ<dim> mapping;
    Table<1,double> rheology_properties;
    unsigned int number_of_visEl, timestep_number;
    TimerOutput computing_timer;
    bool rebuild_matrix_sparsity;

    //Periodic
    std::vector<typename DoFHandler<dim>::active_face_iterator> FaceMap;
    std::vector<Point<dim> > FaceMap_cenPos;
    std::vector<unsigned int> FaceMap_FaceNo;
    std::vector<std::pair<unsigned int, unsigned int> > Periodic_FacePair;
    
    //Mesh
    bool iGmsh;
    std::string input_mesh_file;
    unsigned int init_level, max_level, min_level, local_ref_no;
    double xmin, xmax, ymin, ymax, zmin, zmax;
    bool is_coarse_mesh;
    double small_ref_rad, big_ref_rad, big_asp_ratio;
    double h_min;
    
    //Particle
    bool is_immobile_particle;
    double factor_for_safeguard;
    bool is_solid_particle;
    unsigned int num_pars;
    bool is_random_particles;
    double par_rad;
    double FacPar;
    double a_raTStre;
    double asp_rat_cylin;
    double ori_x, ori_y, ori_z;
    std::vector<Point<dim> > cenPar, image_cenPar;
    std::vector<bool> is_solid_inside;
    std::vector<double> viscosity_distb;
    Point<dim> orient_vector;
    double jeff_orb, tau_nonsp, alpha_factor;
    std::vector<double> twoDimn_ang_vel;
    std::vector<double> avrU, avrV, avrW;
    unsigned int what_viscous_method; 
    double num_ele_cover_particle;
    
    //Equation
    double dt;
    unsigned int endCycle;
    double totVis;
    double tauD;
    double beta;
    double shrF;
    double osi_amplitude;
    double osi_ang_freq;
    double imposed_vel_at_wall;
    
    //Viscoelastic Flow
    std::vector<typename DoFHandler<dim>::active_cell_iterator>  
 periodic_cells_stress;
    std::vector<double> visEls_bnd_value;
    
    //Restart
    unsigned int index_for_restart;
    unsigned int restart_no_timestep;
    bool is_restart;

    
    //Problem
    unsigned int dimn;
    bool reaDat;
    bool initial_mesh_check;
    unsigned int strt_Cycle;
    bool is_recover_triangulation;
    bool is_recover_orientation;
    bool is_periodic;
    double error_stokes;
    double error_stress;
    unsigned int type_model;
    double model_parameter;
    bool is_debug_mode;
    unsigned int output_fac;
    unsigned int refine_fac;
    
    //index
    unsigned int dx, dy, dz;
    unsigned int x, y, z;
    unsigned int inU, inV, inW, inP;
    unsigned int inT11, inT22, inT12, inT33, inT13, inT23;
    
};

template <int dim>
ViscoElasticFlow<dim>::ViscoElasticFlow (ParameterHandler &param)
    :
    trilinos_communicator (Utilities::Trilinos::comm_world()),
    pcout (std::cout, Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)==0),
    prm(param),
    stokes_fe (FE_Q<dim>(2), dim ,FE_Q<dim>(1), 1),
    stress_fe (FE_DGQ<dim>(2), 3*(dim-1)),
    fe_levelset (1),
    stokes_dof_handler (triangulation),
    stress_dof_handler (triangulation),
    dof_handler_levelset (triangulation),
    mapping (2),
    rheology_properties(20),
    number_of_visEl (3*(dim-1)),
    timestep_number (0),
    computing_timer (pcout, TimerOutput::summary,TimerOutput::wall_times),
    rebuild_matrix_sparsity (false),
    is_coarse_mesh (true),
    is_solid_particle (false),
    is_solid_inside (1000000),
    viscosity_distb (1000000),
    visEls_bnd_value (3*(dim-1))
{}

template <int dim>
ViscoElasticFlow<dim>::~ViscoElasticFlow ()
{
    stokes_dof_handler.clear ();
    stress_dof_handler.clear ();
    dof_handler_levelset.clear ();
}

template <int dim>
void ViscoElasticFlow<dim>::create_triangulation ()
{
    pcout << "* Create Mesh Information..." << std::endl;

    const Point<dim> p1 = ( (dim == 2)? (Point<dim> (xmin, ymin)) :
      (Point<dim> (xmin, ymin, zmin)) );
    
    const Point<dim> p2 = ( (dim == 2)? (Point<dim> (xmax, ymax)) :
      (Point<dim> (xmax, ymax, zmax)) );

    if (iGmsh == false)
    {
      GridGenerator::hyper_rectangle (triangulation,p1,p2,false);
      triangulation.refine_global (init_level);
    }
    else
    {
      std::ostringstream input_xx;
      input_xx << "input_meshes/" << input_mesh_file;  
      GridIn<dim> gmsh_input;
      std::ifstream in(input_xx.str().c_str());
      gmsh_input.attach_triangulation (triangulation);
      gmsh_input.read_msh (in);
    }

    std::vector<unsigned int> bnd_min, bnd_max;
    unsigned int dummy = 0;
    for (unsigned int d=0; d<dim; ++d) 
    {bnd_max.push_back(dummy); bnd_min.push_back(dummy);}
    
    unsigned int what_shear_axis = 1;
    if (is_periodic)
    {
      // Naturally, the counter is to assign the pair the number as (1,2),(3,4)
      // Need to modify for shear direction to be given from "input.prm' file
      unsigned int counter_for_bnd = 1;
      for (unsigned int n_axis=0; n_axis<dim; ++n_axis)
      {
 bool not_shear_axis = true; if (n_axis == what_shear_axis) not_shear_axis = false;

 if (not_shear_axis){
   bnd_min[n_axis] = counter_for_bnd; bnd_max[n_axis] = counter_for_bnd + 1;
   counter_for_bnd = counter_for_bnd + 2;}
      }
    }
    
    for (typename Triangulation<dim>::active_cell_iterator
      cell=triangulation.begin_active();
      cell!=triangulation.end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
    if (cell->face(f)->at_boundary())
    {
      cell->face(f)->set_boundary_indicator (6);
 
      const Point<dim> face_center = cell->face(f)->center();

      if (is_periodic)
      for (unsigned int d=0; d<dim; ++d)
      {
 if (d != what_shear_axis)
 {
   if (std::abs(face_center[d] - p1[d]) < 1e-6)
     cell->face(f)->set_boundary_indicator (bnd_min[d]);
                
   if (std::abs(face_center[d] - p2[d]) < 1e-6)
     cell->face(f)->set_boundary_indicator (bnd_max[d]);
 }
      }
    }
}

template <int dim>
void ViscoElasticFlow<dim>::initial_refined_coarse_mesh ()
{
  pcout << "* Initial Refined Mesh... ";
  double range = double(xmax-xmin)*0.0625;
  pcout << range << std::endl;
  
  for (unsigned int i=0; i<(max_level-1-init_level); ++i)    
  {
    for (typename Triangulation<dim>::active_cell_iterator
      cell=triangulation.begin_active();
      cell!=triangulation.end(); ++cell)
    {
 cell->clear_coarsen_flag ();
 cell->clear_refine_flag ();
 
 Point<dim> c = cell->center();
 
 if (dim == 2)
 if (  (std::abs(c[0]) - range) < 1e-8 &&
  (std::abs(c[1]) - range) < 1e-8)
 {
              cell->clear_coarsen_flag();
              cell->set_refine_flag();
        }
        
 if (dim == 3)
 if (  (std::abs(c[0]) - range) < 1e-8 &&
  (std::abs(c[1]) - range) < 1e-8 &&
  (std::abs(c[2]) - range) < 1e-8)
 {
              cell->clear_coarsen_flag();
              cell->set_refine_flag();
        }
        
    }
    triangulation.execute_coarsening_and_refinement ();
    pcout << triangulation.n_active_cells() << " | ";
  }
  pcout << std::endl;
}

template <int dim>
void ViscoElasticFlow<dim>::setup_dofs (bool rebuild_matrix_sparsity)
{
   pcout << "* Setup Dofs..." << std::endl;
   std::vector<unsigned int> stokes_sub_blocks (dim+1, 0);
   stokes_sub_blocks[dim] = 1;

   std::vector<unsigned int> stress_sub_blocks (number_of_visEl, 0);
   for (unsigned int i = 0 ; i<number_of_visEl; ++i)
    stress_sub_blocks[i] = i;

   GridTools::partition_triangulation (Utilities::Trilinos::get_n_mpi_processes(trilinos_communicator),
                                      triangulation);

   {
     stokes_dof_handler.distribute_dofs (stokes_fe);
     DoFRenumbering::subdomain_wise (stokes_dof_handler);
     DoFRenumbering::component_wise (stokes_dof_handler, stokes_sub_blocks);

     stokes_constraints.clear ();
     DoFTools::make_hanging_node_constraints (stokes_dof_handler,
                                              stokes_constraints);
     
//   std::vector<bool> velocity_profile_u(dim+1,true);
//   typename DoFHandler<dim>::active_cell_iterator  
//       initial_cell = stokes_dof_handler.begin_active();
//   unsigned int dof_number_pressure = initial_cell->vertex_dof_index(0 , dim);
//   boundary_values[dof_number_pressure] = 0.0;
//   velocity_profile_u[dim] = false;

  FEValuesExtractors::Vector velocity_components(0);
  
  VectorTools::interpolate_boundary_values (  stokes_dof_handler,
                                              6,
                                              Stokes_BoundaryValues<dim>(imposed_vel_at_wall),
                                              stokes_constraints,
                                              stokes_fe.component_mask(velocity_components));

//   if (Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)==1)
//   {
//     stokes_constraints.add_line  (dof_number_pressure);
//     stokes_constraints.add_entry  (dof_number_pressure, dof_number_pressure, 1.0);
//     stokes_constraints.set_inhomogeneity (dof_number_pressure, 0.0);
//   }      
  
     if (is_periodic && is_coarse_mesh) 
     {
        make_periodicity_constraints ( stokes_dof_handler,
     1,
     2,
     0,
     stokes_constraints);
 if (dim == 3)
   make_periodicity_constraints ( stokes_dof_handler,
       3,
       4,
       2,
       stokes_constraints);
 is_coarse_mesh = true;
     }
     stokes_constraints.close ();
   }

   {
     stress_dof_handler.distribute_dofs (stress_fe);
     DoFRenumbering::subdomain_wise (stress_dof_handler);
     DoFRenumbering::component_wise (stress_dof_handler, stress_sub_blocks);
   }

   std::vector<unsigned int> stokes_dofs_per_block (2);
   DoFTools::count_dofs_per_block (stokes_dof_handler, stokes_dofs_per_block,
                                   stokes_sub_blocks);

   std::vector<unsigned int> stress_dofs_per_block (number_of_visEl);
   DoFTools::count_dofs_per_block (stress_dof_handler, stress_dofs_per_block,
                                   stress_sub_blocks);

   unsigned int n_u = stokes_dofs_per_block[0];
   unsigned int n_p = stokes_dofs_per_block[1];
   unsigned int n_s = stress_dofs_per_block[0];
   
   if (tauD < 1e-10)
    pcout << "* Elem. = " << triangulation.n_active_cells() << std::endl;
   else 
    pcout << "* Elem. = " << triangulation.n_active_cells() << std::endl;
   
   {
     stokes_partitioner.clear();
     std::vector<unsigned int> local_dofs (dim+1);
     DoFTools::
       count_dofs_with_subdomain_association (stokes_dof_handler,
                                              Utilities::Trilinos::get_this_mpi_process(trilinos_communicator),
                                              local_dofs);
     unsigned int n_local_velocities = 0;
     for (unsigned int c=0; c<dim; ++c)
       n_local_velocities += local_dofs[c];

     const unsigned int n_local_pressures = local_dofs[dim];

     Epetra_Map map_u(-1, n_local_velocities, 0, trilinos_communicator);
     stokes_partitioner.push_back (map_u);
     Epetra_Map map_p(-1, n_local_pressures, 0, trilinos_communicator);
     stokes_partitioner.push_back (map_p);
   }
   
   {
     stress_partitioner.clear();
     std::vector<unsigned int> local_dofs (number_of_visEl);
     DoFTools::
       count_dofs_with_subdomain_association (stress_dof_handler,
                                              Utilities::Trilinos::get_this_mpi_process(trilinos_communicator),
                                              local_dofs);
     unsigned int n_local_stresses = 0;
     for (unsigned int c=0; c<number_of_visEl; ++c)
       n_local_stresses += local_dofs[c];

     Epetra_Map map_s(-1, local_dofs[0], 0, trilinos_communicator);
     for (unsigned int i=0; i<number_of_visEl ; ++i)
       stress_partitioner.push_back (map_s);
   }
 
   {
     dof_handler_levelset.distribute_dofs (fe_levelset);
     DoFRenumbering::subdomain_wise (dof_handler_levelset);

     constraint_levelset.clear ();
     DoFTools::make_hanging_node_constraints (dof_handler_levelset,
                                              constraint_levelset);
     
     unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
    Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
     Epetra_Map map_i (-1, local_dofs, 0, trilinos_communicator);
     particle_pos.reinit (map_i);
     viscosity_solution.reinit (map_i);
     levelset_solution.reinit (map_i);
     old_levelset_solution.reinit (map_i);
     level_set_normal_x.reinit (map_i);
     level_set_normal_y.reinit (map_i);
     level_set_normal_z.reinit (map_i);
   }
   

   if (rebuild_matrix_sparsity)
   {
     {
 stokes_matrix.clear ();
 TrilinosWrappers::BlockSparsityPattern sp (stokes_partitioner);
 Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);

 for (unsigned int c=0; c<dim+1; ++c)
        for (unsigned int d=0; d<dim+1; ++d)
          if (! ((c==dim) && (d==dim)))
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;
   
 DoFTools::make_sparsity_pattern (stokes_dof_handler, coupling, sp,
      stokes_constraints, false,
      Utilities::Trilinos::
      get_this_mpi_process(trilinos_communicator));
       sp.compress();
       std::cout << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)
                 <<  " th For Stokes System" << " "
                 << sp.n_nonzero_elements()  << std::endl;
       stokes_matrix.reinit (sp);
     }

     {
       Amg_preconditioner.reset ();
       Mp_preconditioner.reset ();
       stokes_preconditioner_matrix.clear ();
       TrilinosWrappers::BlockSparsityPattern sp (stokes_partitioner);
       Table<2,DoFTools::Coupling> coupling (dim+1, dim+1);
       for (unsigned int c=0; c<dim+1; ++c)
         for (unsigned int d=0; d<dim+1; ++d)
          if (c == d)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;
       DoFTools::make_sparsity_pattern (stokes_dof_handler, coupling, sp,
                                        stokes_constraints, false,
       Utilities::Trilinos::
                                        get_this_mpi_process(trilinos_communicator));
       sp.compress();
       std::cout << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)
                 <<  " th For Precon System" << " "
                 << sp.n_nonzero_elements()  << std::endl;

       stokes_preconditioner_matrix.reinit (sp);
     }
     
     if (tauD > 1.0e-11)
     {
       if (is_periodic == true) make_peri_cell_stress ();
       T_preconditioner.reset ();
       TrilinosWrappers::BlockSparsityPattern sp (stress_partitioner);
       Table<2,DoFTools::Coupling> coupling (3*(dim-1), 3*(dim-1));
       coupling[0][0] = DoFTools::always;
       make_flux_sparsity_pattern (stress_dof_handler, sp,
                                   coupling , coupling,
                                   is_periodic,
                                   periodic_cells_stress,
                                   Utilities::Trilinos::
                                   get_this_mpi_process(trilinos_communicator));
       sp.compress();
       std::cout << Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)
                 <<  " th For Stress System" << " "
                 << sp.n_nonzero_elements()  << std::endl;
       stress_matrix.reinit (sp);
     }

     {
 matrix_levelset.clear();
 unsigned int local_dofs = DoFTools::count_dofs_with_subdomain_association (dof_handler_levelset,
      Utilities::Trilinos::get_this_mpi_process(trilinos_communicator));
 Epetra_Map map_i (-1, local_dofs, 0, trilinos_communicator);
     
 TrilinosWrappers::SparsityPattern sp (map_i);
 DoFTools::make_sparsity_pattern ( dof_handler_levelset, sp,
       constraint_levelset, false,
       Utilities::Trilinos::
       get_this_mpi_process(trilinos_communicator));
 sp.compress();
 matrix_levelset.reinit (sp);
     }     
   }

   stokes_solution.reinit (stokes_partitioner);
   stokes_old_solution.reinit (stokes_partitioner);
   stokes_rhs.reinit (stokes_partitioner);
   stress_solution.reinit (stress_partitioner);
   stress_old_solution.reinit (stress_partitioner);
   stress_rhs.reinit (stress_partitioner);
}


template <int dim>
void ViscoElasticFlow<dim>::build_stokes_preconditioner ()
{
  pcout << "* Build Stokes Preconditioner..." << std::endl;
  
  assemble_stokes_preconditioner ();

  Mp_preconditioner  = std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionILU>
                       (new TrilinosWrappers::PreconditionILU());
  Amg_preconditioner = std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG>
   (new TrilinosWrappers::PreconditionAMG());

  std::vector<std::vector<bool> > constant_modes;
  std::vector<bool>  velocity_components (dim+1,true);
  velocity_components[dim] = false;
  
  DoFTools::extract_constant_modes (stokes_dof_handler, 
          velocity_components, 
          constant_modes);

  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.constant_modes = constant_modes;
  amg_data.elliptic = true;
  amg_data.higher_order_elements = true;
  amg_data.smoother_sweeps = 2;
  amg_data.aggregation_threshold = 0.02;

  Mp_preconditioner->initialize (stokes_preconditioner_matrix.block(1,1));
  Amg_preconditioner->initialize(stokes_preconditioner_matrix.block(0,0),
      amg_data);
}

template <int dim>
void ViscoElasticFlow<dim>::assemble_stokes_preconditioner ()
{
  pcout << "* Assemble Stokes Preconditioner..." << std::endl;
  
  stokes_preconditioner_matrix = 0;

  const QGauss<dim> quadrature_formula (3);
  FEValues<dim> fe_values (stokes_fe, 
     quadrature_formula,
                         update_values    |
                         update_quadrature_points  |
                         update_JxW_values |
                         update_gradients);

  FEValues<dim> fe_levelset_values (fe_levelset, quadrature_formula,
          update_values);
  
  const unsigned int dofs_per_cell = stokes_fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  std::vector<Tensor<2,dim> > phi_grad_u (dofs_per_cell);
  std::vector<double> phi_p (dofs_per_cell);
  std::vector<SymmetricTensor<2,dim> > grads_phi_u (dofs_per_cell);
  std::vector<double> viscosity_values (dofs_per_cell);
  
  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  typename DoFHandler<dim>::active_cell_iterator
   cell = stokes_dof_handler.begin_active(),
   endc = stokes_dof_handler.end();

  typename DoFHandler<dim>::active_cell_iterator
   interface_cell = dof_handler_levelset.begin_active();
 
  unsigned int counter_cell=0;
  for (; cell!=endc; ++cell, ++interface_cell, ++counter_cell)
  if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
  {
    fe_values.reinit (cell);
    fe_levelset_values.reinit (interface_cell);
    
    if (what_viscous_method == 1) 
      fe_levelset_values.get_function_values(viscosity_solution, viscosity_values);
    
    cell->get_dof_indices (local_dof_indices);
    local_matrix = 0;
    
    for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
 phi_grad_u[k] = fe_values[velocities].gradient(k,q);
 grads_phi_u[k] = fe_values[velocities].symmetric_gradient(k,q);
 phi_p[k] = fe_values[pressure].value(k, q);
      }

      double visco_xx = 0.0;
      if (what_viscous_method == 0) visco_xx = viscosity_distb[counter_cell];
      if (what_viscous_method == 1) visco_xx = viscosity_values[q];
      
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        local_matrix(i,j) += (
    visco_xx*
    scalar_product (phi_grad_u[i], phi_grad_u[j])
//     2.0*nu*grads_phi_u[i]*grads_phi_u[j]
    +
    (1./visco_xx) * phi_p[i] * phi_p[j]
//     phi_p[i] * phi_p[j]
         )*
         fe_values.JxW(q);
    }

    stokes_constraints.distribute_local_to_global (local_matrix,
             local_dof_indices,
                  stokes_preconditioner_matrix);
  }

//   std::map<unsigned int,double> boundary_values;
//   std::vector<bool> velocity_profile_u(dim+1,true);
//   typename DoFHandler<dim>::active_cell_iterator  
//       initial_cell = stokes_dof_handler.begin_active();
//   unsigned int dof_number_pressure = initial_cell->vertex_dof_index(0 , dim);
//   boundary_values[dof_number_pressure] = 0.0;
//   velocity_profile_u[dim] = false;

//   VectorTools::interpolate_boundary_values (  stokes_dof_handler,
//                                               6,
//                                               Stokes_BoundaryValues<dim>(imposed_vel_at_wall),
//                                               boundary_values,
//                                               velocity_profile_u);

//   TrilinosWrappers::MPI::BlockVector tmp (stokes_partitioner);

//   MatrixTools::apply_boundary_values (  boundary_values,
//                                         stokes_preconditioner_matrix,
//                                         tmp,
//                                         tmp,
//                                         false);

 stokes_preconditioner_matrix.compress(VectorOperation::add);
}


template <int dim>
void ViscoElasticFlow<dim>::assemble_stokes_system ()
{
  pcout << "* Assemble Stokes Equation..." << std::endl;

  stokes_matrix=0;
  stokes_rhs = 0;

  const QGauss<dim> quadrature_formula(3);

  FEValues<dim> stokes_fe_values (stokes_fe, quadrature_formula,
                                  update_values    |
                                  update_quadrature_points  |
                                  update_JxW_values |
                                  update_gradients);

  FEValues<dim> stress_fe_values (stress_fe, quadrature_formula,
                                  update_values);

  FEValues<dim> fe_levelset_values (fe_levelset, quadrature_formula,
          update_values);
  
  const unsigned int dofs_per_cell = stokes_fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double> local_rhs (dofs_per_cell);
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  std::vector<Vector<double> > solu (n_q_points , Vector<double>(number_of_visEl));
  std::vector<double>          viscosity_values (dofs_per_cell);

  std::vector<Tensor<1,dim> >          phi_u       (dofs_per_cell);
  std::vector<SymmetricTensor<2,dim> > grads_phi_u (dofs_per_cell);
  std::vector<double>                  div_phi_u   (dofs_per_cell);
  std::vector<double>                  phi_p       (dofs_per_cell);

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    endc = stokes_dof_handler.end(),
    stress_cell =  stress_dof_handler.begin_active();

  typename DoFHandler<dim>::active_cell_iterator
   interface_cell = dof_handler_levelset.begin_active();
 
  unsigned int counter_cell = 0;
  for (; cell!=endc; ++cell , ++stress_cell, ++interface_cell, ++counter_cell)
  if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
  {
    stokes_fe_values.reinit (cell);
    stress_fe_values.reinit (stress_cell);
    fe_levelset_values.reinit (interface_cell);
    
    stress_fe_values.get_function_values (stress_solution, solu);
    if (what_viscous_method == 1) fe_levelset_values.get_function_values (viscosity_solution,
               viscosity_values);
    cell->get_dof_indices (local_dof_indices);

    local_matrix = 0; local_rhs = 0;
    
    for (unsigned int q=0; q<n_q_points; ++q)
    {
      
      double visco_xx = 0.0;
      if (what_viscous_method == 0) visco_xx = viscosity_distb[counter_cell];
      if (what_viscous_method == 1) visco_xx = viscosity_values[q];
      
      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
        phi_u[k] = stokes_fe_values[velocities].value (k,q);

        grads_phi_u[k] = stokes_fe_values[velocities].symmetric_gradient(k,q);
        div_phi_u[k]   = stokes_fe_values[velocities].divergence (k, q);
        phi_p[k]       = stokes_fe_values[pressure].value (k, q);
      }

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        local_matrix(i,j) += ( 2.0*
    visco_xx*
    grads_phi_u[i]*grads_phi_u[j]
                              - div_phi_u[i] * phi_p[j]
                              - phi_p[i] * div_phi_u[j])
                              * stokes_fe_values.JxW(q);
    }

    for (unsigned int q=0 ; q < n_q_points ; ++q)
    for (unsigned int i=0 ; i < dofs_per_cell ;++i)
    {
      const unsigned int comp_i = stokes_fe.system_to_component_index (i).first;

      if(comp_i == inU)
      {
        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inU] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT11);

        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inV] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT12);

        if (dim == 3)
        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inW] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT13);
      };

      if(comp_i == inV)
      {
        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inU] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT12);

        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inV] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT22);

        if (dim == 3)
        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inW] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT23);
      };

      if (dim == 3 && comp_i == inW)
      {
        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inU] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT13);

        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inV] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT23);

        local_rhs(i) -= stokes_fe_values.shape_grad (i,q)[inW] *
                        stokes_fe_values.JxW(q) *
                        solu[q](inT33);
      };
    };

    stokes_constraints.distribute_local_to_global ( local_matrix,
            local_rhs,
            local_dof_indices,
            stokes_matrix,
            stokes_rhs);
  } //cell,mpi

//   std::map<unsigned int,double> boundary_values;
//   std::vector<bool> velocity_profile_u(dim+1,true);
//   typename DoFHandler<dim>::active_cell_iterator  
//       initial_cell = stokes_dof_handler.begin_active();
//   unsigned int dof_number_pressure = initial_cell->vertex_dof_index(0 , dim);
//   boundary_values[dof_number_pressure] = 0.0;
//   velocity_profile_u[dim] = false;

//   VectorTools::interpolate_boundary_values (  stokes_dof_handler,
//                                               6,
//                                               Stokes_BoundaryValues<dim>(imposed_vel_at_wall),
//                                               boundary_values,
//                                               velocity_profile_u);

//   TrilinosWrappers::MPI::BlockVector
//     distributed_stokes_solution (stokes_partitioner);
//   distributed_stokes_solution = stokes_solution;

//   MatrixTools::apply_boundary_values (  boundary_values,
//                                         stokes_matrix,
//                                         distributed_stokes_solution,
//                                         stokes_rhs,
//                                         false);

 stokes_matrix.compress(VectorOperation::add);
 stokes_rhs.compress(VectorOperation::add);
}


template <int dim>
void ViscoElasticFlow<dim>::solve (unsigned int part, unsigned int ist)
{
  pcout << "* Solve " << part << " and " << ist << "... "; 

  if (part == 0)
  {
    const LinearSolvers::BlockSchurPreconditioner<TrilinosWrappers::PreconditionAMG,
                                                  TrilinosWrappers::PreconditionILU>
    preconditioner (stokes_matrix, *Mp_preconditioner, *Amg_preconditioner);

    SolverControl solver_control (stokes_matrix.m(),
                                  1e-7*stokes_rhs.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::BlockVector>
      gmres (solver_control,SolverGMRES<TrilinosWrappers::MPI::BlockVector >::AdditionalData(100));

    TrilinosWrappers::MPI::BlockVector 
      distributed_stokes_solution (stokes_partitioner);
    distributed_stokes_solution = stokes_solution;

    const unsigned int start = 
      distributed_stokes_solution.block(1).local_range().first + 
      distributed_stokes_solution.block(0).size();
    const unsigned int end = 
      distributed_stokes_solution.block(1).local_range().second + 
      distributed_stokes_solution.block(0).size();
    for (unsigned int i=start; i<end; ++i)
      if (stokes_constraints.is_constrained (i))
        distributed_stokes_solution(i) = 0;
      
    gmres.solve(stokes_matrix, distributed_stokes_solution, stokes_rhs,
      preconditioner);

    stokes_solution = distributed_stokes_solution;

    pcout << solver_control.last_step() << std::endl;

    stokes_constraints.distribute (stokes_solution);
    
  }

  if (part == 1)
  {
      SolverControl solver_control (stress_matrix.block(0,0).m(),
                                  error_stress*stress_rhs.l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector>
 gmres (solver_control,SolverGMRES<TrilinosWrappers::MPI::Vector >::AdditionalData(100));

      TrilinosWrappers::MPI::BlockVector
 distributed_stress_solution (stress_partitioner);
    
      gmres.solve (stress_matrix.block(0,0), distributed_stress_solution.block(ist),
    stress_rhs.block(ist), *T_preconditioner);
    
      stress_solution.block(ist) = distributed_stress_solution.block(ist);
      pcout << solver_control.last_step() << std::endl;
  }
}

template <int dim>
void ViscoElasticFlow<dim>::viscosity_distribution ()
{
  pcout << "* Viscosity Distribution.. " << std::endl;
  unsigned int counter_cell = 0;
  std::vector<unsigned int> local_dof_indices (fe_levelset.dofs_per_cell);
  typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler_levelset.begin_active(),
   endc = dof_handler_levelset.end();
 
  num_ele_cover_particle = 0.0;
  
  for (; cell!=endc; ++cell, ++counter_cell)
//   if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
  {
    Point<dim> c = cell->center();
    cell->get_dof_indices (local_dof_indices);

    std::pair<unsigned int,double> distant_of_par = distant_from_particles (c, par_rad, a_raTStre);
    viscosity_distb[counter_cell] = totVis*(1.0-beta);
    
    if (distant_of_par.second <0.0 && num_pars > 0) 
    {
      for (unsigned int k=0; k<fe_levelset.dofs_per_cell; ++k)
 particle_pos(local_dof_indices[k]) = 1.0;
      
      is_solid_inside[counter_cell] = true; 
      viscosity_distb[counter_cell] = totVis*(1.0-beta)*FacPar;
      num_ele_cover_particle = num_ele_cover_particle + 1.0; 
    }
  }
  
  std::vector<Point<dim> > sp (dof_handler_levelset.n_dofs());
  MappingQ<dim> ff(fe_levelset.get_degree());
  DoFTools::map_dofs_to_support_points (ff, dof_handler_levelset, sp);
  for (unsigned int i=0; i<viscosity_solution.size(); ++i)
  {
    std::pair<unsigned int,double> 
      distant_of_par = distant_from_particles (sp[i], par_rad, a_raTStre);   
      
    viscosity_solution[i] = totVis*(1.0-beta);
    if (distant_of_par.second < 0.0 && num_pars > 0)
      viscosity_solution[i] = totVis*(1.0-beta)*FacPar;
  }
}

template <int dim>
std::pair<unsigned int, double>
ViscoElasticFlow<dim>::distant_from_particles (Point<dim>  &coor, 
       double  particle_radius,
       double  a_raTStre11)
{
  unsigned int q1 = std::numeric_limits<unsigned int>::max();
  double q2 = std::numeric_limits<double>::max();
  
  if (std::abs(a_raTStre11 - 0.0) <= 1e-6)
  {
    for (unsigned int n = 0 ; n < num_pars ; ++n)
    {
      double tt = cenPar[n].distance(coor) - particle_radius;
      double qq = image_cenPar[n].distance(coor) - particle_radius;
      if (std::min (tt, qq) < q2)
 {q1 = n; q2 = std::min(std::min (tt, qq), q2);}
    } 
  } 
  else
  {
    q1 = 0;
    
    Vector<double> x (dim), a(dim);

    x(0) = coor[0];
    x(1) = coor[1];
    if (dim == 3) x(2) = coor[2];

    //would get the xMx^(T)
    {
 FullMatrix<double> M(dim, dim), Idn(dim, dim), aPP(dim, dim);
 for (unsigned int i = 0; i < dim; ++i)
 for (unsigned int j = 0; j < dim; ++j)
 {
     if (i == j)
  Idn(i, i) = 1.0;

     aPP (i, j) = a_raTStre11*orient_vector(i)*orient_vector(j);

     M (i, j) = Idn (i, j) + aPP (i, j);
 }

 M.vmult (a, x);
 
 double b = 0.0;
 for (unsigned int d=0; d<dim; ++d)
   b += a(d)*x(d);
 
 //min_tq = b - particle_radius*particle_radius;
 q2 = particle_radius - std::sqrt(b);
 q2 = -q2;

 if (q2 < 0 && std::abs(asp_rat_cylin)>0)
 {
     double tt = 0.0;
     for (unsigned int d=0; d<dim; ++d)
       tt += x[d]*orient_vector(d);

     if (tt < -particle_radius*asp_rat_cylin || 
  tt > particle_radius*asp_rat_cylin)
       q2 = -q2;
 }
    }
  }
  return std::make_pair(q1, q2); 
}


template <int dim>
void ViscoElasticFlow<dim>::make_periodicity_constraints (DoFHandler<dim> &dof_handler,
       types::boundary_id   b_id1,
       types::boundary_id   b_id2,
       int                  direction,
       ConstraintMatrix &constraint_matrix)
{
//     pcout << "* Make Periodicity... " << static_cast<unsigned int> (b_id1) << " | " 
//           << static_cast<unsigned int> (b_id2) << std::endl;
    typedef typename DoFHandler<dim>::face_iterator FaceIterator;
    typedef std::map<FaceIterator, std::pair<FaceIterator, std::bitset<3> > > FaceMap;
    Tensor<1, dim> offset;

    std::set<typename DoFHandler<dim>::face_iterator> faces1;
    std::set<typename DoFHandler<dim>::face_iterator> faces2;

    for (typename DoFHandler<dim>::cell_iterator 
   cell = dof_handler.begin();
   cell != dof_handler.end(); ++cell)
      {
        for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
          {
            const typename DoFHandler<dim>::face_iterator face = cell->face(i);

            if (face->at_boundary() && face->boundary_indicator() == b_id1)
     {
//        pcout << "1 = " << face->center() << std::endl;
              faces1.insert(face);
     }
     

            if (face->at_boundary() && face->boundary_indicator() == b_id2)
     {
//        pcout << "2 = " << face->center() << std::endl;
              faces2.insert(face);
     }
          }
      }

//     Assert (faces1.size() == faces2.size(),
//             ExcMessage ("Unmatched faces on periodic boundaries"));
  
  typedef std::pair<FaceIterator, std::bitset<3> > ResultPair;
  std::map<FaceIterator, ResultPair> matched_faces;
    
    // Match with a complexity of O(n^2). This could be improved...
    std::bitset<3> orientation;
    
//     pcout << "* Match Faces... " << std::endl;
    typedef typename std::set<FaceIterator>::const_iterator SetIterator;
    for (SetIterator it1 = faces1.begin(); it1 != faces1.end(); ++it1)
      {
        for (SetIterator it2 = faces2.begin(); it2 != faces2.end(); ++it2)
          {
            if (GridTools::orthogonal_equality(orientation, *it1, *it2,
                                               direction, offset))
              {
//   pcout << *it1 << " | " << *it2 << std::endl;
                // We have a match, so insert the matching pairs and
                // remove the matched cell in faces2 to speed up the
                // matching:
                matched_faces[*it1] = std::make_pair(*it2, orientation);
                faces2.erase(it2);
                break;
              }
          }
      }

//       pcout << "* Insert Constraints... " << std::endl;
      FEValuesExtractors::Vector velocities(0);
      const ComponentMask component_mask = stokes_fe.component_mask (velocities);
      
      for (typename FaceMap::iterator it = matched_faces.begin();
         it != matched_faces.end(); ++it)
      {
        typedef typename DoFHandler<dim>::face_iterator FaceIterator;
        const FaceIterator &face_1 = it->first;
        const FaceIterator &face_2 = it->second.first;
        const std::bitset<3> &orientation = it->second.second;

        Assert(face_1->at_boundary() && face_2->at_boundary(),
               ExcInternalError());

        Assert (face_1->boundary_indicator() == b_id1 &&
                face_2->boundary_indicator() == b_id2,
                ExcInternalError());

        Assert (face_1 != face_2,
                ExcInternalError());

//  pcout << face_1 << std::endl;
 make_periodicity_constraints (face_1,
          face_2,
          constraint_matrix,
          component_mask,
          orientation[0],
          orientation[1],
          orientation[2]);
 
 
      }
}

template <int dim>
void ViscoElasticFlow<dim>::make_periodicity_constraints (
      const typename DoFHandler<dim>::face_iterator    &face_1,
      const typename identity<typename DoFHandler<dim>::face_iterator>::type &face_2,
      ConstraintMatrix  &constraint_matrix,
      const ComponentMask  &component_mask,
      const bool   face_orientation,
      const bool   face_flip,
      const bool   face_rotation)
{
//      static const int dim = FaceIterator::AccessorType::dimension;

     Assert( (dim != 1) ||
            (face_orientation == true &&
             face_flip == false &&
             face_rotation == false),
            ExcMessage ("The supplied orientation "
                        "(face_orientation, face_flip, face_rotation) "
                        "is invalid for 1D"));

     Assert( (dim != 2) ||
            (face_orientation == true &&
             face_rotation == false),
            ExcMessage ("The supplied orientation "
                        "(face_orientation, face_flip, face_rotation) "
                        "is invalid for 2D"));

     Assert(face_1 != face_2,
     ExcMessage ("face_1 and face_2 are equal! Cannot constrain DoFs "
                       "on the very same face"));

     Assert(face_1->at_boundary() && face_2->at_boundary(),
     ExcMessage ("Faces for periodicity constraints must be on the boundary"));


     // A lookup table on how to go through the child faces depending on the
     // orientation:

     static const int lookup_table_2d[2][2] =
     {
       //          flip:
       {0, 1}, //  false
       {1, 0}, //  true
     };

     static const int lookup_table_3d[2][2][2][4] =
     {
       //                    orientation flip  rotation
       { { {0, 2, 1, 3}, //  false       false false
    {2, 3, 0, 1}, //  false       false true
  },
  { {3, 1, 2, 0}, //  false       true  false
    {1, 0, 3, 2}, //  false       true  true
  },
       },
       { { {0, 1, 2, 3}, //  true        false false
    {1, 3, 0, 2}, //  true        false true
  },
  { {3, 2, 1, 0}, //  true        true  false
  {2, 0, 3, 1}, //  true        true  true
  },
       },
     };

   // In the case that both faces have children, we loop over all
   // children and apply make_periodicty_constrains recursively:
   if (face_1->has_children() && face_2->has_children())
   {
     Assert(face_1->n_children() == GeometryInfo<dim>::max_children_per_face &&
    face_2->n_children() == GeometryInfo<dim>::max_children_per_face,
    ExcNotImplemented());

     for (unsigned int i = 0; i < GeometryInfo<dim>::max_children_per_face; ++i)
     {
  // Lookup the index for the second face
  unsigned int j;
  switch (dim)
  {
    case 2:
    j = lookup_table_2d[face_flip][i];
    break;
    case 3:
    j = lookup_table_3d[face_orientation][face_flip][face_rotation][i];
    break;
    default:
    AssertThrow(false, ExcNotImplemented());
  }

  make_periodicity_constraints (face_1->child(i),
      face_2->child(j),
      constraint_matrix,
      component_mask,
      face_orientation,
      face_flip,
      face_rotation);
     }
   }
 else
   // otherwise at least one of the two faces is active and
   // we need to enter the constraints
   {
     if (face_2->has_children() == false)
     {
//        pcout << "Case 1" << std::endl;
       set_periodicity_constraints(face_2, face_1,
                                      FullMatrix<double>(IdentityMatrix(face_1->get_fe(face_1->nth_active_fe_index(0)).dofs_per_face)),
                                      constraint_matrix,
                                      component_mask,
                                      face_orientation, face_flip, face_rotation);
     }
     else
     {
//        pcout << "Case 2" << std::endl;
       set_periodicity_constraints(face_1, face_2,
                                      FullMatrix<double>(IdentityMatrix(face_2->get_fe(face_2->nth_active_fe_index(0)).dofs_per_face)),
                                      constraint_matrix,
                                      component_mask,
                                      face_orientation, face_flip, face_rotation);
     }
   }
}
template <int dim>
void ViscoElasticFlow<dim>::set_periodicity_constraints (
        const typename DoFHandler<dim>::face_iterator                     &face_1,
        const typename identity<typename DoFHandler<dim>::face_iterator>::type &face_2,
        const FullMatrix<double>                        &transformation,
        ConstraintMatrix                         &constraint_matrix,
        const ComponentMask                             &component_mask,
        const bool                                       face_orientation,
        const bool                                       face_flip,
        const bool                                       face_rotation)
{

//       pcout << "set_periodicity_constraints..1" << std::endl;
//       static const int dim      = FaceIterator::AccessorType::dimension;
      static const int spacedim = dim;

      // we should be in the case where face_1 is active, i.e. has no children:
      Assert (!face_1->has_children(),
              ExcInternalError());

      Assert (face_1->n_active_fe_indices() == 1,
              ExcInternalError());

      // if face_2 does have children, then we need to iterate over them
    if (face_2->has_children()) 
    {
//       pcout << "ddd0" << std::endl;
      Assert (face_2->n_children() == GeometryInfo<dim>::max_children_per_face,
       ExcNotImplemented());
      const unsigned int dofs_per_face
 = face_1->get_fe(face_1->nth_active_fe_index(0)).dofs_per_face;
      FullMatrix<double> child_transformation (dofs_per_face, dofs_per_face);
      FullMatrix<double> subface_interpolation (dofs_per_face, dofs_per_face);
      for (unsigned int c=0; c<face_2->n_children(); ++c)
      {
              // get the interpolation matrix recursively from the one that
              // interpolated from face_1 to face_2 by multiplying from the
              // left with the one that interpolates from face_2 to
              // its child
   face_1->get_fe(face_1->nth_active_fe_index(0))
   .get_subface_interpolation_matrix (face_1->get_fe(face_1->nth_active_fe_index(0)),
           c,
           subface_interpolation);
   subface_interpolation.mmult (child_transformation, transformation);
   set_periodicity_constraints(face_1, face_2->child(c),
          child_transformation,
          constraint_matrix, component_mask,
          face_orientation, face_flip, face_rotation);
       
      }
//       pcout << "ddd1" << std::endl;
    }
    else
        // both faces are active. we need to match the corresponding DoFs of both faces
    {
//       pcout << "eee0" << std::endl;
      const unsigned int face_1_index = face_1->nth_active_fe_index(0);
      const unsigned int face_2_index = face_2->nth_active_fe_index(0);
          Assert(face_1->get_fe(face_1_index) == face_2->get_fe(face_1_index),
                 ExcMessage ("Matching periodic cells need to use the same finite element"));

      const FiniteElement<dim> &fe = face_1->get_fe(face_1_index);

          Assert(component_mask.represents_n_components(fe.n_components()),
                 ExcMessage ("The number of components in the mask has to be either "
                             "zero or equal to the number of components in the finite " "element."));

      const unsigned int dofs_per_face = fe.dofs_per_face;

      std::vector<types::global_dof_index> dofs_1(dofs_per_face);
      std::vector<types::global_dof_index> dofs_2(dofs_per_face);

      face_1->get_dof_indices(dofs_1, face_1_index);
      face_2->get_dof_indices(dofs_2, face_2_index);

      for (unsigned int i=0; i < dofs_per_face; i++)
      {
 if (dofs_1[i] == numbers::invalid_dof_index ||
     dofs_2[i] == numbers::invalid_dof_index)
 {
                  /* If either of these faces have no indices, stop.  This is so
                   * that there is no attempt to match artificial cells of
                   * parallel distributed triangulations.
                   *
                   * While it seems like we ought to be able to avoid even calling
                   * set_periodicity_constraints for artificial faces, this
                   * situation can arise when a face that is being made periodic
                   * is only partially touched by the local subdomain.
                   * make_periodicity_constraints will be called recursively even
                   * for the section of the face that is not touched by the local
                   * subdomain.
                   *
                   * Until there is a better way to determine if the cells that
                   * neighbor a face are artificial, we simply test to see if the
                   * face does not have a valid dof initialization.
                   */
     return;
 }
      }

          // Well, this is a hack:
          //
          // There is no
          //   face_to_face_index(face_index,
          //                      face_orientation,
          //                      face_flip,
          //                      face_rotation)
          // function in FiniteElementData, so we have to use
          //   face_to_cell_index(face_index, face
          //                      face_orientation,
          //                      face_flip,
          //                      face_rotation)
          // But this will give us an index on a cell - something we cannot work
          // with directly. But luckily we can match them back :-]

      std::map<unsigned int, unsigned int> cell_to_rotated_face_index;

          // Build up a cell to face index for face_2:
      for (unsigned int i = 0; i < dofs_per_face; ++i)
      {
 const unsigned int cell_index = fe.face_to_cell_index(i, 0, 
             /* It doesn't really matter, just assume
                                                            * we're on the first face...
                                                            */
                                                             true, false, false // default orientation
                                                             );
        cell_to_rotated_face_index[cell_index] = i;
      }

    // pcout << "set_periodicity_constraints..2" << std::endl;
          // loop over all dofs on face 2 and constrain them again the ones on face 1
      for (unsigned int i=0; i<dofs_per_face; ++i)
      if (!constraint_matrix.is_constrained(dofs_2[i]))
      if ((component_mask.n_selected_components(fe.n_components())
           == fe.n_components())
           ||
           component_mask[fe.face_system_to_component_index(i).first])
      {
                  // as mentioned in the comment above this function, we need
                  // to be careful about treating identity constraints differently.
                  // consequently, find out whether this dof 'i' will be
                  // identity constrained
                  //
                  // to check whether this is the case, first see whether there are
                  // any weights other than 0 and 1, then in a first stage make sure
                  // that if so there is only one weight equal to 1
   bool is_identity_constrained = true;
   for (unsigned int jj=0; jj<dofs_per_face; ++jj)
   if (((transformation(i,jj) == 0) || (transformation(i,jj) == 1)) == false)
   {
     is_identity_constrained = false;
     break;
   }
          unsigned int identity_constraint_target = numbers::invalid_unsigned_int;
          if (is_identity_constrained == true)
          {
     bool one_identity_found = false;
     for (unsigned int jj=0; jj<dofs_per_face; ++jj)
     if (transformation(i,jj) == 1)
     {
       if (one_identity_found == false)
       {
  one_identity_found = true;
  identity_constraint_target = jj;
       }
       else
       {
  is_identity_constrained = false;
  identity_constraint_target = numbers::invalid_unsigned_int;
  break;
       }
     }
   }

                  // now treat constraints, either as an equality constraint or
                  // as a sequence of constraints
          if (is_identity_constrained == true)
          {
                      // Query the correct face_index on face_2 respecting the given
                      // orientation:
                      const unsigned int j =
                        cell_to_rotated_face_index[fe.face_to_cell_index(identity_constraint_target,
                                                                         0, /* It doesn't really matter, just assume
                           * we're on the first face...
                           */
                                                                         face_orientation, face_flip, face_rotation)];

                      // if the two aren't already identity constrained (whichever way
                      // around, then enter the constraint. otherwise there is nothing
                      // for us still to do
          
              
//      const bool ddd = constraint_velocity.are_identity_constrained(dofs_2[i], dofs_1[i]);
//      const ConstraintLine &p = lines[lines_cache[calculate_line_index(dofs_2[i])]];
//           bool dd = constraint_matrix.is_identity_constrained(dofs_2[i]);
//      const ConstraintMatrix::ConstraintLine p = constraint_matrix.lines[constraint_matrix.lines_cache[constraint_matrix.calculate_line_index(dofs_2[i])]];
//      ConstraintMatrix::ConstraintLine p;
//      unsigned int ddd = p.calculate_line_index(dofs_2[i]);
//      unsigned int ddd = constraint_matrix.calculate_line_index(dofs_2[i]);
   
//                       if (constraint_matrix.are_identity_constrained(dofs_2[i], dofs_1[i]) == false)
    if (constraint_matrix.is_identity_constrained(dofs_2[i]) == false &&
      constraint_matrix.is_identity_constrained(dofs_1[i]) == false)
                        {
                          constraint_matrix.add_line(dofs_2[i]);
                          constraint_matrix.add_entry(dofs_2[i], dofs_1[j], 1);
                        }
          }
          else
          {
                      // this is just a regular constraint. enter it piece by piece
                      constraint_matrix.add_line(dofs_2[i]);
                      for (unsigned int jj=0; jj<dofs_per_face; ++jj)
                        {
                          // Query the correct face_index on face_2 respecting the given
                          // orientation:
                          const unsigned int j =
                            cell_to_rotated_face_index[fe.face_to_cell_index(jj, 0, 
    /* It doesn't really matter, just assume
                               * we're on the first face...*/
                                                                             face_orientation, face_flip, face_rotation)];

                          // And finally constrain the two DoFs respecting component_mask:
                          if (transformation(i,jj) != 0)
                            constraint_matrix.add_entry(dofs_2[i], dofs_1[j],
                                                        transformation(i,jj));
                        }
           }
      } //loop_constraint_matrix
//    pcout << "eee1" << std::endl;
    } //if(face_2->child?)
} 


template <int dim>
void ViscoElasticFlow<dim>::save_snapshot (unsigned int np)
{
  pcout << "* Save Snapshot..." << std::endl;
  std::ostringstream filename_mesh;
  filename_mesh << "store_mesh/remesh_" << Utilities::int_to_string(np, 4)<< ".mesh";
  std::ofstream output_mesh (filename_mesh.str().c_str());
  
  boost::archive::text_oarchive oa(output_mesh);
  triangulation.save(oa, 1); 
  
  output_mesh.close ();
    
  {
    BlockVector<double> y_stokes(stokes_solution);
    y_stokes = stokes_solution;
    std::stringstream filename;
    filename << "store_solution/stokes.solution";
    std::ofstream output (filename.str().c_str());
    output.precision (14);
    y_stokes.block_write (output);
    
    if (tauD > 1e-11)
    {
      BlockVector<double> y_stress(stress_solution);
      y_stress = stress_solution;
      std::stringstream filename;
      filename << "store_solution/stress.solution";
      std::ofstream output (filename.str().c_str());
      output.precision (14);
      y_stress.block_write (output);      
    }
  }
}

template <int dim>
void ViscoElasticFlow<dim>::resume_snapshot ()
{
  pcout << "* Resume Snapshot..." << std::endl;
  std::ostringstream filename_mesh;
  filename_mesh << "store_mesh/remesh.mesh";
  std::ifstream output_mesh (filename_mesh.str().c_str());
  
  boost::archive::text_iarchive ia(output_mesh);
  triangulation.load(ia, 1); 
  
  setup_dofs(true);
    
  {
    BlockVector<double> y_stokes(stokes_solution);
    y_stokes = stokes_solution;
    std::ostringstream filename;
    filename << "store_solution/stokes.solution";
    std::ifstream output (filename.str().c_str());
    output.precision (14);
    y_stokes.block_read (output);
    stokes_solution = y_stokes;
    
    if (tauD > 1e-11)
    {
      BlockVector<double> y_stress(stress_solution);
      y_stress = stress_solution;
      std::ostringstream filename;
      filename << "store_solution/stress.solution";
      std::ifstream output (filename.str().c_str());
      output.precision (14);
      y_stress.block_read (output); 
      stress_solution = y_stress;
    }
  }
}

template <int dim>
void ViscoElasticFlow<dim>::plotting_solution (unsigned int np)  const
{
  pcout <<"* Print Solutions..." << std::endl;

  const FESystem<dim> joint_fe (stokes_fe, 1, 
      stress_fe, 1, 
      fe_levelset, 1);

  DoFHandler<dim> joint_dof_handler (triangulation);
  joint_dof_handler.distribute_dofs (joint_fe);
  Assert (joint_dof_handler.n_dofs() ==
          stokes_dof_handler.n_dofs() +
          stress_dof_handler.n_dofs() +
   dof_handler_levelset.n_dofs (),
          ExcInternalError());

  Vector<double> joint_solution (joint_dof_handler.n_dofs());

  if (Utilities::Trilinos::get_this_mpi_process(trilinos_communicator) == 0)
    {
      {
 std::vector<unsigned int> local_joint_dof_indices (joint_fe.dofs_per_cell);
 std::vector<unsigned int> local_stokes_dof_indices (stokes_fe.dofs_per_cell);
 std::vector<unsigned int> local_stress_dof_indices (stress_fe.dofs_per_cell);
 std::vector<unsigned int> local_interface_dof_indices (fe_levelset.dofs_per_cell);
 
        typename DoFHandler<dim>::active_cell_iterator
          joint_cell       = joint_dof_handler.begin_active(),
          joint_endc       = joint_dof_handler.end(),
          stokes_cell      = stokes_dof_handler.begin_active(),
          stress_cell      = stress_dof_handler.begin_active(),
          interface_cell   = dof_handler_levelset.begin_active ();

        for (; joint_cell!=joint_endc; ++joint_cell, ++stokes_cell, ++stress_cell, ++interface_cell)
          {
            joint_cell->get_dof_indices (local_joint_dof_indices);
            stokes_cell->get_dof_indices (local_stokes_dof_indices);
            stress_cell->get_dof_indices (local_stress_dof_indices);
     interface_cell->get_dof_indices (local_interface_dof_indices);

            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
              if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                  joint_solution(local_joint_dof_indices[i])
                    = stokes_solution(local_stokes_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
              else if (joint_fe.system_to_base_index(i).first.first == 1)
                {
                  joint_solution(local_joint_dof_indices[i])
                    = stress_solution(local_stress_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
              else if (joint_fe.system_to_base_index(i).first.first == 2)
                {
                  joint_solution(local_joint_dof_indices[i])
                    = particle_pos(local_interface_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
          }
      }

      std::vector<std::string> joint_solution_names (dim, "V");
      joint_solution_names.push_back ("P");
      joint_solution_names.push_back ("Txx");
      joint_solution_names.push_back ("Tyy");
      joint_solution_names.push_back ("Txy");
      if (dim == 3)
      {
 joint_solution_names.push_back ("Tzz");
 joint_solution_names.push_back ("Txz");
 joint_solution_names.push_back ("Tyz");
      }
      joint_solution_names.push_back ("Par");
      
      DataOut<dim> data_out;
      data_out.attach_dof_handler (joint_dof_handler);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation
      (dim + 1 + 3*(dim-1) + 1, DataComponentInterpretation::component_is_scalar);
      for (unsigned int i=0; i<dim; ++i)
        data_component_interpretation[i]
        = DataComponentInterpretation::component_is_part_of_vector;

      data_out.add_data_vector (joint_solution, joint_solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);

      data_out.build_patches (1);

      std::ostringstream filename;
      filename << "vtu_files/s" << Utilities::int_to_string(np, 4) << ".vtu";

      std::ofstream output (filename.str().c_str());
      data_out.write_vtu (output);
    }
}


template <int dim>
void ViscoElasticFlow<dim>::make_flux_sparsity_pattern (
      DoFHandler<dim> &dof,
      TrilinosWrappers::BlockSparsityPattern &sparsity,
      const Table<2,DoFTools::Coupling> &int_mask,
      const Table<2,DoFTools::Coupling> &flux_mask,
      bool is_periodic,
      std::vector<typename DoFHandler<dim>::active_cell_iterator> &peri_cells,
      const unsigned int subdomain_id)
{
  const unsigned int n_dofs = dof.n_dofs();
  const FiniteElement<dim> &fe = dof.get_fe();
  const unsigned int n_comp = fe.n_components();

  Assert (sparsity.n_rows() == n_dofs,
   ExcDimensionMismatch (sparsity.n_rows(), n_dofs));
  Assert (sparsity.n_cols() == n_dofs,
   ExcDimensionMismatch (sparsity.n_cols(), n_dofs));
  Assert (int_mask.n_rows() == n_comp,
   ExcDimensionMismatch (int_mask.n_rows(), n_comp));
  Assert (int_mask.n_cols() == n_comp,
   ExcDimensionMismatch (int_mask.n_cols(), n_comp));
  Assert (flux_mask.n_rows() == n_comp,
   ExcDimensionMismatch (flux_mask.n_rows(), n_comp));
  Assert (flux_mask.n_cols() == n_comp,
   ExcDimensionMismatch (flux_mask.n_cols(), n_comp));

  const unsigned int total_dofs = fe.dofs_per_cell;
  std::vector<unsigned int> dofs_on_this_cell(total_dofs);
  std::vector<unsigned int> dofs_on_other_cell(total_dofs);
  Table<2,bool> support_on_face(
    total_dofs, GeometryInfo<dim>::faces_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active(),
        endc = dof.end();

  const Table<2,DoFTools::Coupling>
    int_dof_mask  = DoFTools::dof_couplings_from_component_couplings(fe, int_mask),
    flux_dof_mask = DoFTools::dof_couplings_from_component_couplings(fe, flux_mask);

  for (unsigned int i=0; i<total_dofs; ++i)
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell;++f)
      support_on_face(i,f) = fe.has_support_on_face(i,f);


  std::vector<bool> user_flags;
  dof.get_tria().save_user_flags(user_flags);
  const_cast<Triangulation<dim> &>(dof.get_tria()).clear_user_flags ();

  unsigned int cell_no = 0;
  for (; cell!=endc; ++cell, ++cell_no)
  if (subdomain_id == cell->subdomain_id())
    {
      cell->get_dof_indices (dofs_on_this_cell);

      for (unsigned int i=0; i<total_dofs; ++i)
 for (unsigned int j=0; j<total_dofs; ++j)
   if (int_dof_mask(i,j) != DoFTools::none)
     sparsity.add (dofs_on_this_cell[i],
     dofs_on_this_cell[j]);


      for (unsigned int face = 0;
    face < GeometryInfo<dim>::faces_per_cell;
    ++face)
 {
   const typename DoFHandler<dim>::face_iterator
            cell_face = cell->face(face);
   if (cell_face->user_flag_set ())
     continue;

   if (cell->at_boundary (face) )
     {
              if (is_periodic == true && cell->face(face)->boundary_indicator() > 0)
              {
                typename DoFHandler<dim>::active_cell_iterator neighbor;
                neighbor = peri_cells[cell_no];
                neighbor->get_dof_indices (dofs_on_other_cell);


                for (unsigned int i=0; i<total_dofs; ++i)
                for (unsigned int j=0; j<total_dofs; ++j)
                {
                  if (flux_dof_mask(i,j) == DoFTools::always)
                  {
                    sparsity.add (dofs_on_this_cell[i],
                                  dofs_on_other_cell[j]);
                    sparsity.add (dofs_on_other_cell[i],
                                  dofs_on_this_cell[j]);
                    sparsity.add (dofs_on_this_cell[i],
                                  dofs_on_this_cell[j]);
                    sparsity.add (dofs_on_other_cell[i],
                                 dofs_on_other_cell[j]);
                  }

                  if (flux_dof_mask(j,i) == DoFTools::always)
                  {
                    sparsity.add (dofs_on_this_cell[j],
                                  dofs_on_other_cell[i]);
                    sparsity.add (dofs_on_other_cell[j],
                                  dofs_on_this_cell[i]);
                    sparsity.add (dofs_on_this_cell[j],
                                  dofs_on_this_cell[i]);
                    sparsity.add (dofs_on_other_cell[j],
                                  dofs_on_other_cell[i]);
                  }
                }
              }
              else if (is_periodic == false || cell->face(face)->boundary_indicator() == 0)
              {
         for (unsigned int i=0; i<total_dofs; ++i)
  {
    const bool i_non_zero_i = support_on_face (i, face);
    for (unsigned int j=0; j<total_dofs; ++j)
      {
        const bool j_non_zero_i = support_on_face (j, face);

        if (flux_dof_mask(i,j) == DoFTools::always)
                        sparsity.add (dofs_on_this_cell[i],
                                      dofs_on_this_cell[j]);
        if (flux_dof_mask(i,j) == DoFTools::nonzero
     && i_non_zero_i && j_non_zero_i)
   sparsity.add (dofs_on_this_cell[i],
          dofs_on_this_cell[j]);
      }
  }
              }
     }
   else
     {
       typename DoFHandler<dim>::cell_iterator
  neighbor = cell->neighbor(face);

       if (cell->neighbor_is_coarser(face))
  continue;

       typename DoFHandler<dim>::face_iterator cell_face = cell->face(face);
       const unsigned int
                neighbor_face = cell->neighbor_of_neighbor(face);

       if (cell_face->has_children())
  {
    for (unsigned int sub_nr = 0;
         sub_nr != cell_face->n_children();
         ++sub_nr)
      {
        const typename DoFHandler<dim>::cell_iterator
                        sub_neighbor
   = cell->neighbor_child_on_subface (face, sub_nr);

        sub_neighbor->get_dof_indices (dofs_on_other_cell);
        for (unsigned int i=0; i<total_dofs; ++i)
   {
     const bool i_non_zero_i = support_on_face (i, face);
     const bool i_non_zero_e = support_on_face (i, neighbor_face);
     for (unsigned int j=0; j<total_dofs; ++j)
       {
         const bool j_non_zero_i = support_on_face (j, face);
         const bool j_non_zero_e  =support_on_face (j, neighbor_face);
         if (flux_dof_mask(i,j) == DoFTools::always)
    {
      sparsity.add (dofs_on_this_cell[i],
      dofs_on_other_cell[j]);
      sparsity.add (dofs_on_other_cell[i],
      dofs_on_this_cell[j]);
      sparsity.add (dofs_on_this_cell[i],
      dofs_on_this_cell[j]);
      sparsity.add (dofs_on_other_cell[i],
      dofs_on_other_cell[j]);
    }
         if (flux_dof_mask(i,j) == DoFTools::nonzero)
    {
      if (i_non_zero_i && j_non_zero_e)
        sparsity.add (dofs_on_this_cell[i],
        dofs_on_other_cell[j]);
      if (i_non_zero_e && j_non_zero_i)
        sparsity.add (dofs_on_other_cell[i],
        dofs_on_this_cell[j]);
      if (i_non_zero_i && j_non_zero_i)
        sparsity.add (dofs_on_this_cell[i],
        dofs_on_this_cell[j]);
      if (i_non_zero_e && j_non_zero_e)
        sparsity.add (dofs_on_other_cell[i],
        dofs_on_other_cell[j]);
    }

         if (flux_dof_mask(j,i) == DoFTools::always)
    {
      sparsity.add (dofs_on_this_cell[j],
      dofs_on_other_cell[i]);
      sparsity.add (dofs_on_other_cell[j],
      dofs_on_this_cell[i]);
      sparsity.add (dofs_on_this_cell[j],
      dofs_on_this_cell[i]);
      sparsity.add (dofs_on_other_cell[j],
      dofs_on_other_cell[i]);
    }
         if (flux_dof_mask(j,i) == DoFTools::nonzero)
    {
      if (j_non_zero_i && i_non_zero_e)
        sparsity.add (dofs_on_this_cell[j],
        dofs_on_other_cell[i]);
      if (j_non_zero_e && i_non_zero_i)
        sparsity.add (dofs_on_other_cell[j],
        dofs_on_this_cell[i]);
      if (j_non_zero_i && i_non_zero_i)
        sparsity.add (dofs_on_this_cell[j],
        dofs_on_this_cell[i]);
      if (j_non_zero_e && i_non_zero_e)
        sparsity.add (dofs_on_other_cell[j],
        dofs_on_other_cell[i]);
    }
       }
   }
        sub_neighbor->face(neighbor_face)->set_user_flag ();
      }
  }
              else
                {
    neighbor->get_dof_indices (dofs_on_other_cell);
    for (unsigned int i=0; i<total_dofs; ++i)
      {
        const bool i_non_zero_i = support_on_face (i, face);
        const bool i_non_zero_e = support_on_face (i, neighbor_face);
        for (unsigned int j=0; j<total_dofs; ++j)
   {
     const bool j_non_zero_i = support_on_face (j, face);
     const bool j_non_zero_e = support_on_face (j, neighbor_face);
     if (flux_dof_mask(i,j) == DoFTools::always)
       {
         sparsity.add (dofs_on_this_cell[i],
         dofs_on_other_cell[j]);
         sparsity.add (dofs_on_other_cell[i],
         dofs_on_this_cell[j]);
         sparsity.add (dofs_on_this_cell[i],
         dofs_on_this_cell[j]);
         sparsity.add (dofs_on_other_cell[i],
         dofs_on_other_cell[j]);
       }
     if (flux_dof_mask(i,j) == DoFTools::nonzero)
       {
         if (i_non_zero_i && j_non_zero_e)
    sparsity.add (dofs_on_this_cell[i],
           dofs_on_other_cell[j]);
         if (i_non_zero_e && j_non_zero_i)
    sparsity.add (dofs_on_other_cell[i],
           dofs_on_this_cell[j]);
         if (i_non_zero_i && j_non_zero_i)
    sparsity.add (dofs_on_this_cell[i],
           dofs_on_this_cell[j]);
         if (i_non_zero_e && j_non_zero_e)
    sparsity.add (dofs_on_other_cell[i],
           dofs_on_other_cell[j]);
       }

     if (flux_dof_mask(j,i) == DoFTools::always)
       {
         sparsity.add (dofs_on_this_cell[j],
         dofs_on_other_cell[i]);
         sparsity.add (dofs_on_other_cell[j],
         dofs_on_this_cell[i]);
         sparsity.add (dofs_on_this_cell[j],
         dofs_on_this_cell[i]);
         sparsity.add (dofs_on_other_cell[j],
         dofs_on_other_cell[i]);
       }
     if (flux_dof_mask(j,i) == DoFTools::nonzero)
       {
         if (j_non_zero_i && i_non_zero_e)
    sparsity.add (dofs_on_this_cell[j],
           dofs_on_other_cell[i]);
         if (j_non_zero_e && i_non_zero_i)
    sparsity.add (dofs_on_other_cell[j],
           dofs_on_this_cell[i]);
         if (j_non_zero_i && i_non_zero_i)
    sparsity.add (dofs_on_this_cell[j],
           dofs_on_this_cell[i]);
         if (j_non_zero_e && i_non_zero_e)
    sparsity.add (dofs_on_other_cell[j],
           dofs_on_other_cell[i]);
       }
   }
      }
    neighbor->face(neighbor_face)->set_user_flag ();
  }
     }
 }
    }


  const_cast<Triangulation<dim> &>(dof.get_tria()).load_user_flags(user_flags);
}

template <int dim>
double Mij ( const FEValuesBase<dim> &fe_values,
  const unsigned int i,
  const unsigned int j,
  const unsigned int q)
{
return  fe_values.shape_value (i , q) *
  fe_values.shape_value (j , q) *
  fe_values.JxW (q);
}

template <int dim>
double Ri ( const FEValuesBase<dim> &fe_values,
  const unsigned int i,
  const unsigned int q)
{
return  fe_values.shape_value (i , q) *
  fe_values.JxW (q);
}


template <int dim>
void ViscoElasticFlow<dim>::make_peri_cell_stress ()
{
  typename DoFHandler<dim>::active_cell_iterator
          cell = stress_dof_handler.begin_active();

  for (unsigned int i=0; i < stress_dof_handler.get_tria().n_active_cells() ; ++i)
    periodic_cells_stress.push_back(cell);

  {
    typename DoFHandler<dim>::active_cell_iterator
      cell0 = stress_dof_handler.begin_active(),
      endc0 = stress_dof_handler.end();

    unsigned int cell_no0 = 0;
    unsigned int cell_no1 = 0;

    bool go1, go2;

    Point<dim> cell0_center , cell1_center;

    for (;cell0!=endc0; ++cell0, ++cell_no0)
    {
      periodic_cells_stress[cell_no0] = cell0;

      go1 = false;
      go2 = false;
      Point<dim> coor = cell0->center();
      cell0_center = coor;

      if (cell0->face(1)->boundary_indicator() == 2)
      {
        typename DoFHandler<dim>::active_cell_iterator
          cell1 = stress_dof_handler.begin_active(),
          endc1 = stress_dof_handler.end();

        cell_no1 = 0;
        for (;cell1!=endc1; ++cell1, ++cell_no1)
        {
          Point<dim> coor = cell1->center();
          cell1_center = coor;

          if (cell1->face(0)->boundary_indicator() == 1)
          {
            if (dim == 2)
            if (std::abs(cell0_center[1] - cell1_center[1]) < 1.0e-10)
            {
              periodic_cells_stress[cell_no0] = cell1;
              periodic_cells_stress[cell_no1] = cell0;
            }

            if (dim == 3)
            if (std::abs(cell0_center[1] - cell1_center[1]) < 1.0e-10 && std::abs(cell0_center[2] - cell1_center[2]) < 1.0e-10)
            {
              periodic_cells_stress[cell_no0] = cell1;
              periodic_cells_stress[cell_no1] = cell0;
            }
          }
        }
      }
      if (dim == 3)
      if (cell0->face(5)->boundary_indicator() == 4)
      {
        typename DoFHandler<dim>::active_cell_iterator
          cell1 = stress_dof_handler.begin_active(),
          endc1 = stress_dof_handler.end();

        cell_no1 = 0;
        for (;cell1!=endc1; ++cell1, ++cell_no1)
        {
          Point<dim> coor = cell1->center();
          cell1_center = coor;

          if (cell1->face(4)->boundary_indicator() == 3)
          {
            if (std::abs(cell0_center[0] - cell1_center[0]) < 1.0e-10 && std::abs(cell0_center[1] - cell1_center[1]) < 1.0e-10)
            {
              periodic_cells_stress[cell_no0] = cell1;
              periodic_cells_stress[cell_no1] = cell0;
            }
          }
        }
      }
    }
  }
}

template <int dim>
void ViscoElasticFlow<dim>::assemble_peri_DGConvect ()
{
  pcout << "* Periodic DG Convective Term..." << std::endl;

  stress_matrix = 0;
  double dt1= 1/dt;

  QGauss<dim>  quadrature(3);
  QGauss<dim-1> face_quadrature(3);

  const unsigned int n_q_points = quadrature.size();
  const unsigned int n_q_points_face = face_quadrature.size();

  const unsigned int dofs_per_cell = stress_dof_handler.get_fe().dofs_per_cell;
  std::vector<unsigned int> dofs (dofs_per_cell);
  std::vector<unsigned int> dofs_neighbor (dofs_per_cell);

  const UpdateFlags update_flags = update_values
                                   | update_gradients
                                   | update_quadrature_points
                                   | update_JxW_values;

  const UpdateFlags face_update_flags = update_values
                                        | update_quadrature_points
                                        | update_JxW_values
                                        | update_normal_vectors;

  const UpdateFlags neighbor_face_update_flags = update_values;

  FEValues<dim> stress_fe_v (
 mapping, stress_fe, quadrature, update_flags);
  FEFaceValues<dim> stress_fe_v_face (
 mapping, stress_fe, face_quadrature, face_update_flags);
  FESubfaceValues<dim> stress_fe_v_subface (
 mapping, stress_fe, face_quadrature, face_update_flags);
  FEFaceValues<dim> stress_fe_v_face_neighbor (
 mapping, stress_fe, face_quadrature, neighbor_face_update_flags);
  FESubfaceValues<dim> stress_fe_v_subface_neighbor (
 mapping, stress_fe, face_quadrature, neighbor_face_update_flags);

  FEValues<dim> stokes_fe_v (
        mapping, stokes_fe, quadrature, update_flags);
  FEFaceValues<dim> stokes_fe_v_face (
        mapping, stokes_fe, face_quadrature, face_update_flags);
  FESubfaceValues<dim> stokes_fe_v_subface (
        mapping, stokes_fe, face_quadrature, face_update_flags);
  FEFaceValues<dim> stokes_fe_v_face_neighbor (
        mapping, stokes_fe, face_quadrature, neighbor_face_update_flags);
  FESubfaceValues<dim> stokes_fe_v_subface_neighbor (
        mapping, stokes_fe, face_quadrature, neighbor_face_update_flags);

  FullMatrix<double> ui_vi_matrix (dofs_per_cell, dofs_per_cell);
  FullMatrix<double> ue_vi_matrix (dofs_per_cell, dofs_per_cell);

  Vector<double>  cell_vector (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
   cell = stress_dof_handler.begin_active(),
   endc = stress_dof_handler.end(),
   stokes_cell = stokes_dof_handler.begin_active();

  std::vector<Vector<double> > stokes_solu (n_q_points , Vector<double>(dim+1));
  std::vector<Vector<double> > stokes_solu_face (n_q_points_face , Vector<double>(dim+1));

  unsigned int cell_no = 0;
  for (;cell!=endc; ++cell, ++stokes_cell, ++cell_no)
  if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
  {
    ui_vi_matrix = 0;
    cell_vector = 0;

    stress_fe_v.reinit (cell);
    stokes_fe_v.reinit (stokes_cell);

    stokes_fe_v.get_function_values (stokes_solution, stokes_solu);
    cell->get_dof_indices (dofs);

    const std::vector<double> &JxW = stress_fe_v.get_JxW_values ();
    const FiniteElement<dim> &feT = stress_fe_v.get_fe ();

    for (unsigned int q=0; q<stress_fe_v.n_quadrature_points; ++q)
    for (unsigned int i=0; i<stress_fe_v.dofs_per_cell; ++i)
    for (unsigned int j=0; j<stress_fe_v.dofs_per_cell; ++j)
    {
      const unsigned int comp_i = feT.system_to_component_index (i).first;
      const unsigned int comp_j = feT.system_to_component_index (j).first;

      if (comp_i == comp_j && comp_i == inT11)
      {

        ui_vi_matrix(i,j) += dt1 *
        stress_fe_v.shape_value(i,q) *
        stress_fe_v.shape_value(j,q) *
        JxW[q];

        for (unsigned d = 0 ; d < dim ; ++d)
          ui_vi_matrix(i,j) -= stress_fe_v.shape_grad(i,q)[d] *
          stokes_solu[q](d) *
          stress_fe_v.shape_value(j,q) *
          JxW[q];
      }
    }

    cell->get_dof_indices (dofs);
    for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
    {
      typename DoFHandler<dim>::face_iterator face=cell->face(face_no);
      typename DoFHandler<dim>::face_iterator stokes_face=stokes_cell->face(face_no);

      ue_vi_matrix = 0;

       if (face->at_boundary())
       {
         typename DoFHandler<dim>::active_cell_iterator neighbor;
         neighbor = periodic_cells_stress[cell_no];
         stress_fe_v_face.reinit (cell, face_no);

         unsigned int np = 0;
         if (face->boundary_indicator() == 2) np = 0;
         if (face->boundary_indicator() == 1) np = 1;

         if (dim == 3)
         {
           if (face->boundary_indicator() == 4) np = 4;
           if (face->boundary_indicator() == 3) np = 5;
         }

         stress_fe_v_face_neighbor.reinit (neighbor, np);

         stokes_fe_v_face.reinit (stokes_cell, face_no);
         stokes_fe_v_face.get_function_values (stokes_solution, stokes_solu_face);

         const std::vector<double> &JxW = stress_fe_v_face.get_JxW_values ();
         const std::vector<Point<dim> > &normals = stress_fe_v_face.get_normal_vectors ();
         const FiniteElement<dim> &fe = stress_fe_v_face.get_fe ();
         const FiniteElement<dim> &fe0 = stress_fe_v_face_neighbor.get_fe ();

         for (unsigned int q=0; q<stress_fe_v_face.n_quadrature_points; ++q)
         {
           double ndotv;
           ndotv = 0.0;
           for (unsigned ii = 0 ; ii < dim ; ++ii)
             ndotv += stokes_solu_face[q](ii) * normals[q](ii);
 
 
           if (ndotv > 0)
           {
      for (unsigned int i=0; i<stress_fe_v_face.dofs_per_cell; ++i)
      for (unsigned int j=0; j<stress_fe_v_face.dofs_per_cell; ++j)
             {
         const unsigned int comp_i = fe.system_to_component_index (i).first;
               const unsigned int comp_j = fe.system_to_component_index (j).first;

               if (comp_i == comp_j && comp_i == inT11)
          ui_vi_matrix(i,j) += ndotv *
          stress_fe_v_face.shape_value(i,q) *
          stress_fe_v_face.shape_value(j,q) *
          JxW[q];
             }
           }
           else if (ndotv < 0 && is_periodic == true)
           {
             for (unsigned int i=0; i<stress_fe_v_face.dofs_per_cell; ++i)
             for (unsigned int j=0; j<stress_fe_v_face_neighbor.dofs_per_cell; ++j)
             {
               unsigned int comp_i=  fe.system_to_component_index (i).first;
               unsigned int comp_j=  fe0.system_to_component_index (j).first;
 
               if (comp_i == comp_j && comp_i == inT11)
               ue_vi_matrix(i,j) += ndotv *
                                    stress_fe_v_face.shape_value(i,q) *
                                    stress_fe_v_face_neighbor.shape_value(j,q) *
                                    JxW[q];
             }
           }
         }
 
         neighbor->get_dof_indices (dofs_neighbor);
 
         for (unsigned int i=0; i<dofs_per_cell; ++i)
         for (unsigned int k=0; k<dofs_per_cell; ++k)
           stress_matrix.add(dofs[i], dofs_neighbor[k],ue_vi_matrix(i,k));
 
       } // face-at_boundary
//       if (face->at_boundary())
//       {
//         stress_fe_v_face.reinit (cell, face_no);
//  stokes_fe_v_face.reinit (stokes_cell, face_no);
//  stokes_fe_v_face.get_function_values (stokes_solution, stokes_solu_face);

//  const std::vector<double> &JxW = stress_fe_v_face.get_JxW_values ();
//  const std::vector<Point<dim> > &normals = stress_fe_v_face.get_normal_vectors ();
//  const FiniteElement<dim> &feT = stress_fe_v_face.get_fe ();

//  for (unsigned int q=0; q<stress_fe_v_face.n_quadrature_points; ++q)
//  {
//    double ndotv = 0.0;
//    for (unsigned ii = 0 ; ii < dim ; ++ii)
//      ndotv += stokes_solu_face[q](ii) * normals[q](ii);

//    if (ndotv > 0)
//    for (unsigned int i=0; i<stress_fe_v_face.dofs_per_cell; ++i)
//    for (unsigned int j=0; j<stress_fe_v_face.dofs_per_cell; ++j)
//           {
//       const unsigned int comp_i = feT.system_to_component_index (i).first;
//             const unsigned int comp_j = feT.system_to_component_index (j).first;

//             if (comp_i == comp_j && comp_i == inT11)
//        ui_vi_matrix(i,j) += ndotv *
//                      stress_fe_v_face.shape_value(i,q) *
//        stress_fe_v_face.shape_value(j,q) *
//        JxW[q];

//           }
//         }
//       }
      else
      {
        typename DoFHandler<dim>::cell_iterator neighbor = cell->neighbor(face_no);
        typename DoFHandler<dim>::cell_iterator stokes_neighbor = stokes_cell->neighbor(face_no);

 if (face->has_children())
 {
   const unsigned int neighbor2 = cell->neighbor_of_neighbor(face_no);

   for (unsigned int subface_no=0 ;
    subface_no<face->n_children(); ++subface_no)
   {
     typename DoFHandler<dim>::active_cell_iterator
     neighbor_child = cell->neighbor_child_on_subface (face_no, subface_no),
            stokes_neighbor_child = stokes_cell->neighbor_child_on_subface (face_no, subface_no);

     Assert (!neighbor_child->has_children(), ExcInternalError());

     ue_vi_matrix = 0;

     stress_fe_v_subface.reinit (cell, face_no, subface_no);
     stokes_fe_v_subface.reinit (stokes_cell, face_no , subface_no);
     stress_fe_v_face_neighbor.reinit (neighbor_child, neighbor2);
            stokes_fe_v_face_neighbor.reinit (stokes_neighbor_child, neighbor2);

     stokes_fe_v_subface.get_function_values (stokes_solution, stokes_solu_face);

     assemble_face_term( stress_fe_v_subface,
        stress_fe_v_face_neighbor,
        ui_vi_matrix,
        ue_vi_matrix,
                stokes_solu_face);

     neighbor_child->get_dof_indices (dofs_neighbor);

     for (unsigned int i=0; i<dofs_per_cell; ++i)
     for (unsigned int k=0; k<dofs_per_cell; ++k)
              stress_matrix.add(dofs[i], dofs_neighbor[k],ue_vi_matrix(i,k));
   }
 }
 else
 {
   if (neighbor->level() == cell->level())
   {
     const unsigned int neighbor2=cell->neighbor_of_neighbor(face_no);

     stress_fe_v_face.reinit (cell, face_no);
     stress_fe_v_face_neighbor.reinit (neighbor, neighbor2);

            stokes_fe_v_face.reinit (stokes_cell, face_no);
     stokes_fe_v_face.get_function_values (stokes_solution, stokes_solu_face);

     assemble_face_term( stress_fe_v_face,
    stress_fe_v_face_neighbor,
    ui_vi_matrix,
    ue_vi_matrix,
    stokes_solu_face);
   }
   else
   {
     Assert(neighbor->level() < cell->level(), ExcInternalError());

     const std::pair<unsigned int, unsigned int> faceno_subfaceno=
    cell->neighbor_of_coarser_neighbor(face_no);
     const unsigned int neighbor_face_no=faceno_subfaceno.first,
       neighbor_subface_no=faceno_subfaceno.second;

     Assert (neighbor->neighbor_child_on_subface (neighbor_face_no,
                                                         neighbor_subface_no)
             == cell,ExcInternalError());

     stress_fe_v_face.reinit (cell, face_no);
            stokes_fe_v_face.reinit (stokes_cell, face_no);
     stress_fe_v_subface_neighbor.reinit (neighbor, neighbor_face_no,neighbor_subface_no);
     stokes_fe_v_face.get_function_values (stokes_solution, stokes_solu_face);

     assemble_face_term( stress_fe_v_face,
    stress_fe_v_subface_neighbor,
    ui_vi_matrix,
    ue_vi_matrix,
    stokes_solu_face);
   }

   neighbor->get_dof_indices (dofs_neighbor);

   for (unsigned int i=0; i<dofs_per_cell; ++i)
   for (unsigned int k=0; k<dofs_per_cell; ++k)
     stress_matrix.add(dofs[i], dofs_neighbor[k],ue_vi_matrix(i,k));
        }//level-end
      } //face_bnd-else-loop-end
    } // face-end

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    for (unsigned int j=0; j<dofs_per_cell; ++j)
      stress_matrix.add(dofs[i], dofs[j], ui_vi_matrix(i,j));

  } // cell-end

  stress_matrix.compress (VectorOperation::add);
  
  T_preconditioner =  std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG>
                                (new TrilinosWrappers::PreconditionAMG);
  T_preconditioner->initialize (stress_matrix.block(0,0));

}


template <int dim>
void ViscoElasticFlow<dim>::assemble_face_term(
     const FEFaceValuesBase<dim>& fe_v,
     const FEFaceValuesBase<dim>& fe_v_neighbor,
     FullMatrix<double> &ui_vi_matrix,
     FullMatrix<double> &ue_vi_matrix,
     std::vector<Vector<double> > &solu) const
{
  const std::vector<double> &JxW = fe_v.get_JxW_values ();
  const std::vector<Point<dim> > &normals = fe_v.get_normal_vectors ();
  const FiniteElement<dim> &fe = fe_v.get_fe ();
  const FiniteElement<dim> &fe0 = fe_v_neighbor.get_fe ();

  for (unsigned int q=0; q<fe_v.n_quadrature_points; ++q)
  {
    double ndotv;
    ndotv = 0.0;
    for (unsigned ii = 0 ; ii < dim ; ++ii)
      ndotv += solu[q](ii) * normals[q](ii);

    if (ndotv > 0)
    {
      for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
      for (unsigned int j=0; j<fe_v.dofs_per_cell; ++j)
      {
        unsigned int comp_i=  fe.system_to_component_index (i).first;
        unsigned int comp_j=  fe.system_to_component_index (j).first;

        if (comp_i == comp_j && comp_i == inT11)
        ui_vi_matrix(i,j) += ndotv *
        fe_v.shape_value(i,q) *
        fe_v.shape_value(j,q) *
             JxW[q];
      }
    }
    else
    {
      for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
      for (unsigned int j=0; j<fe_v_neighbor.dofs_per_cell; ++j)
      {
        unsigned int comp_i=  fe.system_to_component_index (i).first;
        unsigned int comp_j=  fe0.system_to_component_index (j).first;

        if (comp_i == comp_j && comp_i == inT11)
          ue_vi_matrix(i,j) += ndotv *
          fe_v.shape_value(i,q) *
          fe_v_neighbor.shape_value(j,q) *
          JxW[q];
      }
    }
  }
}

template <int dim>
void ViscoElasticFlow<dim>::assemble_RHSVisEls ()
{

  pcout << "* Assemble RHS..." << std::flush;
  stress_rhs = 0;

  double dt1= 1/dt;
  double tauD1 = 1/tauD;

  QGauss<dim>  quadrature_formula (3);
  QGauss<dim-1> face_quadrature_formula (3);

  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_q_points_face = face_quadrature_formula.size();

  FEValues<dim> stress_fe_values (stress_fe, quadrature_formula,
  UpdateFlags( update_values    |
                               update_gradients |
                                update_q_points  |
                                update_JxW_values));

  FEValues<dim> stokes_fe_values (stokes_fe, quadrature_formula,
  UpdateFlags(    update_values    |
                                update_gradients));

  const unsigned int dofs_per_cell = stress_fe.dofs_per_cell;
  Vector<double> cell_rhs (dofs_per_cell);

  std::vector<unsigned int> local_dof_indices (dofs_per_cell);

  std::vector<std::vector<Tensor<1,dim> > >  stokes_solGrads (n_q_points,std::vector<Tensor<1,dim> > (dim+1));
  std::vector<Vector<double> > stokes_solu_face (n_q_points_face , Vector<double>(dim+1));
  std::vector<Vector<double> > stress_solu (n_q_points , Vector<double>(3*(dim-1)));

  typename DoFHandler<dim>::active_cell_iterator cell = stress_dof_handler.begin_active(),
                                                 endc = stress_dof_handler.end(),
       stokes_cell = stokes_dof_handler.begin_active();

  const UpdateFlags face_update_flags = update_values
                                        | update_quadrature_points
                                        | update_JxW_values
                                        | update_normal_vectors;
  FEFaceValues<dim> stress_fe_v_face (
    mapping, stress_fe, face_quadrature_formula, face_update_flags);
  FEFaceValues<dim> stokes_fe_v_face (
  mapping, stokes_fe, face_quadrature_formula, face_update_flags);

  for (; cell!=endc; ++cell, ++stokes_cell)
  if (cell->subdomain_id() == Utilities::Trilinos::get_this_mpi_process(trilinos_communicator))
  {
    cell->get_dof_indices (local_dof_indices);

    stress_fe_values.reinit (cell);
    stokes_fe_values.reinit (stokes_cell);

    stress_fe_values.get_function_values (stress_solution, stress_solu);
    stokes_fe_values.get_function_gradients (stokes_solution, stokes_solGrads);

    cell_rhs = 0;

    unsigned int in_i = 0;
    double model_para_vispoly = 0.0;
    if (totVis*beta > 0.0) model_para_vispoly = model_parameter/totVis*beta;

    for (unsigned int q=0 ; q < n_q_points ; ++q)
    for (unsigned int i = 0; i < n_q_points ; ++i)
    {
      in_i = stress_fe.component_to_system_index (inT11,i);
      cell_rhs (in_i) += 2.0*totVis*beta*tauD1*Ri(stress_fe_values,in_i,q)*stokes_solGrads[q][inU][dx];
      cell_rhs (in_i) += Ri(stress_fe_values,in_i,q)*
                  (
                         + dt1*stress_solu[q](inT11)
                         - tauD1*stress_solu[q](inT11)
                         + 2*stress_solu[q](inT11)*stokes_solGrads[q][inU][dx]
                         + 2*stress_solu[q](inT12)*stokes_solGrads[q][inU][dy]
                         );

      if (dim == 3)
      cell_rhs (in_i) += Ri(stress_fe_values,in_i,q)*2*stress_solu[q](inT13)*stokes_solGrads[q][inU][dz];


      if (type_model == 1)
      {
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         ( stress_solu[q](inT11)*stress_solu[q](inT11) +
    stress_solu[q](inT12)*stress_solu[q](inT12));

   if (dim == 3)
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         (stress_solu[q](inT13)*stress_solu[q](inT13));

      }

      in_i = stress_fe.component_to_system_index (inT22,i);
      cell_rhs(in_i) += 2.0*totVis*beta*tauD1*Ri(stress_fe_values,in_i,q)*stokes_solGrads[q][inV][dy];
      cell_rhs (in_i) += Ri(stress_fe_values,in_i,q)*
                         (
                         + dt1*stress_solu[q](inT22)
                         - tauD1*stress_solu[q](inT22)
                         + 2*stress_solu[q](inT12)*stokes_solGrads[q][inV][dx]
                         + 2*stress_solu[q](inT22)*stokes_solGrads[q][inV][dy]
                         );

      if (dim == 3)
      cell_rhs (in_i) += Ri(stress_fe_values,in_i,q)*
                         2*stress_solu[q](inT23)*stokes_solGrads[q][inV][dz];

      if (type_model == 1)
      {
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         ( stress_solu[q](inT12)*stress_solu[q](inT12) +
    stress_solu[q](inT22)*stress_solu[q](inT22));

   if (dim == 3)
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         (stress_solu[q](inT23)*stress_solu[q](inT23));
      }

      in_i = stress_fe.component_to_system_index (inT12,i);
      cell_rhs(in_i) += totVis*beta*tauD1*Ri(stress_fe_values,in_i,q)*(stokes_solGrads[q][inV][dx] +
            stokes_solGrads[q][inU][dy] );
      cell_rhs(in_i) += Ri(stress_fe_values,in_i,q)*
   (
   + dt1*stress_solu[q](inT12)
   - tauD1*stress_solu[q](inT12)
   + stress_solu[q](inT12)*stokes_solGrads[q][inU][dx]
   + stress_solu[q](inT22)*stokes_solGrads[q][inU][dy]
                        + stress_solu[q](inT11)*stokes_solGrads[q][inV][dx]
                        + stress_solu[q](inT12)*stokes_solGrads[q][inV][dy]
                        );

      if (dim == 3)
      cell_rhs (in_i) += Ri(stress_fe_values,in_i,q)*
                         (
                         + stress_solu[q](inT23)*stokes_solGrads[q][inU][dz]
                         + stress_solu[q](inT13)*stokes_solGrads[q][inV][dz]
                         );

      if (type_model == 1)
      {
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         ( stress_solu[q](inT11)*stress_solu[q](inT12) +
    stress_solu[q](inT12)*stress_solu[q](inT22));

   if (dim == 3)
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         (stress_solu[q](inT13)*stress_solu[q](inT23));
      }

      if (dim == 3)
      {
        in_i = stress_fe.component_to_system_index (inT33,i);
        cell_rhs(in_i) += 2.0*totVis*beta*tauD1*Ri(stress_fe_values,in_i,q)*stokes_solGrads[q][inW][dz];
        cell_rhs (in_i) += Ri(stress_fe_values,in_i,q)*
                           (
                           + dt1*stress_solu[q](inT33)
                           - tauD1*stress_solu[q](inT33)
                           + 2*stress_solu[q](inT13)*stokes_solGrads[q][inW][dx]
                           + 2*stress_solu[q](inT23)*stokes_solGrads[q][inW][dy]
                    + 2*stress_solu[q](inT33)*stokes_solGrads[q][inW][dz]
             );

   if (type_model == 1)
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         ( stress_solu[q](inT13)*stress_solu[q](inT13) +
    stress_solu[q](inT23)*stress_solu[q](inT23) +
    stress_solu[q](inT33)*stress_solu[q](inT33));

        in_i = stress_fe.component_to_system_index (inT13,i);
 cell_rhs(in_i) += totVis*beta*tauD1*Ri(stress_fe_values,in_i,q)*(stokes_solGrads[q][inW][dx] +
              stokes_solGrads[q][inU][dz]);

 cell_rhs (in_i) += Ri(stress_fe_values,in_i,q)*
                           (
                           + dt1*stress_solu[q](inT13)
                           - tauD1*stress_solu[q](inT13)
                           + stress_solu[q](inT13)*stokes_solGrads[q][inU][dx]
                           + stress_solu[q](inT23)*stokes_solGrads[q][inU][dy]
                           + stress_solu[q](inT11)*stokes_solGrads[q][inW][dx]
                           + stress_solu[q](inT12)*stokes_solGrads[q][inW][dy]
                           + stress_solu[q](inT33)*stokes_solGrads[q][inU][dz]
                           + stress_solu[q](inT13)*stokes_solGrads[q][inW][dz]
      );

   if (type_model == 1)
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         ( stress_solu[q](inT11)*stress_solu[q](inT13) +
    stress_solu[q](inT12)*stress_solu[q](inT23) +
    stress_solu[q](inT13)*stress_solu[q](inT33));

        in_i = stress_fe.component_to_system_index (inT23,i);
        cell_rhs(in_i) += totVis*beta*tauD1*Ri(stress_fe_values,in_i,q)*(stokes_solGrads[q][inW][dy] +
              stokes_solGrads[q][inV][dz]);
 cell_rhs (in_i) += Ri(stress_fe_values,in_i,q)*
                           (
                           + dt1*stress_solu[q](inT23)
                           - tauD1*stress_solu[q](inT23)
                           + stress_solu[q](inT13)*stokes_solGrads[q][inV][dx]
                           + stress_solu[q](inT23)*stokes_solGrads[q][inV][dy]
                           + stress_solu[q](inT12)*stokes_solGrads[q][inW][dx]
                           + stress_solu[q](inT22)*stokes_solGrads[q][inW][dy]
                           + stress_solu[q](inT33)*stokes_solGrads[q][inV][dz]
                           + stress_solu[q](inT23)*stokes_solGrads[q][inW][dz]
      );

   if (type_model == 1)
   cell_rhs (in_i) -= model_para_vispoly*
         Ri(stress_fe_values,in_i,q)*
         ( stress_solu[q](inT12)*stress_solu[q](inT13) +
    stress_solu[q](inT22)*stress_solu[q](inT23) +
    stress_solu[q](inT23)*stress_solu[q](inT33));
      }; // dim=3
    }; // q,i-index

    if (is_periodic == false)
    {
      for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        stress_fe_v_face.reinit (cell, face_no);
        stokes_fe_v_face.reinit (stokes_cell, face_no);
        stokes_fe_v_face.get_function_values (stokes_solution, stokes_solu_face);

        typename DoFHandler<dim>::face_iterator face=cell->face(face_no);

        if (face->at_boundary())
        {
   const std::vector<double> &JxW = stress_fe_v_face.get_JxW_values ();
   const std::vector<Point<dim> > &normals = stress_fe_v_face.get_normal_vectors ();
   const FiniteElement<dim> &feT = stress_fe_v_face.get_fe ();

   std::vector<Vector<double> > g (stress_fe_v_face.n_quadrature_points, 
       Vector<double>(3*(dim-1)));

   Viscoelastic_BoundaryValues<dim> bndfun( dt, 
       shrF, 
       totVis*beta, 
       tauD,
       visEls_bnd_value
          );
   
   bndfun.vector_value_list (stress_fe_v_face.get_quadrature_points(), g);

   for (unsigned int kk=0; kk<3*(dim-1); ++kk) visEls_bnd_value[kk] = g[0](kk);

   for (unsigned int q=0; q<stress_fe_v_face.n_quadrature_points; ++q)
   {
     double ndotv;
     ndotv = 0.0;
     for (unsigned ii = 0 ; ii < dim ; ++ii)
       ndotv += stokes_solu_face[q](ii) * normals[q](ii);

     if (ndotv <= 0)
     for (unsigned int i=0; i<stress_fe_v_face.dofs_per_cell; ++i)
            {
              unsigned int comp_i=  feT.system_to_component_index (i).first;

       cell_rhs(i) -= ndotv *
                             g[q](comp_i) *
                             stress_fe_v_face.shape_value(i,q) *
                             JxW[q];
            }
   }
        } // at_boundary
      } // face_loop
    } //is_periodic

    for (unsigned int i=0 ; i < dofs_per_cell ; ++i)
      stress_rhs (local_dof_indices[i]) += cell_rhs(i);

  };

  stress_rhs.compress (VectorOperation::add);

  pcout << std::endl;

}

template <int dim>
void ViscoElasticFlow<dim>::refine_mesh (bool matrix_init)
{
  pcout <<"* Refinement..."<<std::endl;

  if (std::abs(a_raTStre) < 1e-6) error_indicator_for_isotropic ();
  if (std::abs(a_raTStre) >= 1e-6) error_indicator_for_anisotropic ();
  
  SolutionTransfer<dim,TrilinosWrappers::BlockVector > stokes_transfer (stokes_dof_handler);
  SolutionTransfer<dim,TrilinosWrappers::BlockVector > stress_transfer (stress_dof_handler);

  TrilinosWrappers::BlockVector stokes_transfer_solution (stokes_solution);
  TrilinosWrappers::BlockVector stress_transfer_solution (stress_solution);

  triangulation.prepare_coarsening_and_refinement ();
  stokes_transfer.prepare_for_coarsening_and_refinement (stokes_transfer_solution);
  stress_transfer.prepare_for_coarsening_and_refinement (stress_transfer_solution);

  triangulation.execute_coarsening_and_refinement ();

  setup_dofs (matrix_init);
//   pcout << "* After Refinement = " << triangulation.n_active_cells() << std::endl;

  TrilinosWrappers::BlockVector new_stokes_solution = stokes_solution;
  stokes_transfer.interpolate (stokes_transfer_solution, new_stokes_solution);
  stokes_solution = new_stokes_solution;

  TrilinosWrappers::BlockVector new_stress_solution = stress_solution;
  stress_transfer.interpolate (stress_transfer_solution, new_stress_solution);
  stress_solution = new_stress_solution;
}

template <int dim>
void ViscoElasticFlow<dim>::error_indicator_for_isotropic ()
{
  pcout <<"* Error Indicator for Isotropic..."<<std::endl;
  
  typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    
  double safe_guard_layer = factor_for_safeguard * h_min;
  
  pcout << "* " << h_min << " " << factor_for_safeguard 
 << ", " << max_level << ", " << min_level << std::endl;
  
  for (; cell!=endc; ++cell)
  {
    cell->clear_refine_flag();
    cell->clear_coarsen_flag();

    Point<dim> c = cell->center();

    std::pair<unsigned int,double> distant_of_par = distant_from_particles (c, par_rad, a_raTStre);
    double absolute_distant_from_particle = std::abs(distant_of_par.second); 
    unsigned int cell_level = cell->level();

    {
  double aa = 0.0;
        double bb = 0.0;
        for (unsigned int i=0; i<local_ref_no; ++i)
        {
            aa = bb;
            bb = aa + 2.0* static_cast<double>(i);
            if (i==0) bb = 1;

            if (absolute_distant_from_particle  >  aa*safe_guard_layer && 
  absolute_distant_from_particle <= bb*safe_guard_layer)
            {
                if (cell_level < max_level-i) cell->set_refine_flag();
                if (cell_level > max_level-i) cell->set_coarsen_flag();
                if (cell_level == max_level-i)
                {
      cell->clear_refine_flag();
      cell->clear_coarsen_flag();
                }
            }
        }
        
        //cut_refinement_particle (coor);
 if (dim == 2)
 if ( (std::abs(c[0]) - 0.25001) > 1e-8 || (std::abs(c[1]) - 0.25001) > 1e-8)
 //if (cut_mintq > 0)
 {
   cell->set_coarsen_flag();
   cell->clear_refine_flag();
 }

 if (dim == 3)
 if (  (std::abs(c[0]) - 0.25001) > 1e-8 ||
  (std::abs(c[1]) - 0.25001) > 1e-8 ||
  (std::abs(c[2]) - 0.25001) > 1e-8)
 {
              cell->set_coarsen_flag();
              cell->clear_refine_flag();
        }
        
    }

    if (cell_level == max_level) cell->clear_refine_flag();
    if (cell_level == min_level) cell->clear_coarsen_flag();
  }  
}

template <int dim>
void ViscoElasticFlow<dim>::error_indicator_for_anisotropic ()
{
//   pcout <<"* Error Indicator for Anisotropic..."<<std::endl;

  typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
  
  for (; cell!=endc; ++cell)
  {
    cell->clear_refine_flag();
    cell->clear_coarsen_flag();

    Point<dim> cell_center = cell->center();
    unsigned int cell_level = cell->level();
    std::vector<bool> boolean_inside_particle;
    
    for (unsigned int i=0; i<2; ++i)
    {
      double radius_var = par_rad;
      if (i == 0) radius_var = par_rad + small_ref_rad;
      if (i == 1) radius_var = par_rad + big_ref_rad;
      
      double a_raT_var = a_raTStre;
      if (i == 0) a_raT_var = a_raTStre + 0.0;
      if (i == 1) a_raT_var = a_raTStre + big_asp_ratio;
      
      std::pair<unsigned int,double> 
 distant_of_par = distant_from_particles (cell_center, radius_var, a_raT_var);

      if (distant_of_par.second < 0.0) boolean_inside_particle.push_back (true);
      if (distant_of_par.second >= 0.0) boolean_inside_particle.push_back (false);
    }
                      
    if( boolean_inside_particle[0] == false &&
 boolean_inside_particle[1] == true)
    {
      cell->set_refine_flag();
    } else 
    {
      cell->set_coarsen_flag();
    }
         
    //cut_refinement_particle (coor);
    if(dim == 2)
    if( (std::abs(cell_center[0]) - 0.25001) > 1e-8 ||
 (std::abs(cell_center[1]) - 0.25001) > 1e-8)
    {
      cell->set_coarsen_flag();
      cell->clear_refine_flag();
    }

    if(dim == 3)
    if( (std::abs(cell_center[0]) - 0.25001) > 1e-8 ||
 (std::abs(cell_center[1]) - 0.25001) > 1e-8 ||
 (std::abs(cell_center[2]) - 0.25001) > 1e-8)
    {
      cell->set_coarsen_flag();
      cell->clear_refine_flag();
    }

    if (cell_level == max_level) cell->clear_refine_flag();
    if (cell_level == min_level) cell->clear_coarsen_flag();
  }  
}

template <int dim>
void ViscoElasticFlow<dim>::compute_particle_dyn_properties ( 
        unsigned int iii, 
        std::ofstream &out_a,
        std::ofstream &out_or)
{
  pcout << "* Compute Particle Dyn. Propeties..." << std::endl;

//   for (unsigned int n_pars=0; n_pars < num_pars; ++n_pars)
  unsigned int n_pars=0;
  {
      QGauss<dim>  quadrature_formula(3);
      const unsigned int n_q_points = quadrature_formula.size();

      FEValues<dim> stokes_fe_values ( stokes_fe,quadrature_formula,
     UpdateFlags(update_values    |
     update_gradients |
     update_q_points  |
     update_JxW_values));

      const unsigned int   dofs_per_cell = stokes_fe.dofs_per_cell;
      std::vector<unsigned int> local_dof_indices (dofs_per_cell);

      typename DoFHandler<dim>::active_cell_iterator  cell, endc;

      cell = stokes_dof_handler.begin_active();
      endc = stokes_dof_handler.end();

      std::vector<Vector<double> > stokes_solu (n_q_points , Vector<double>(dim+1));
      std::vector<std::vector<Tensor<1,dim> > > stokes_solGrads ( stokes_fe_values.n_quadrature_points ,
            std::vector<Tensor<1,dim> > (dim+1));

      Vector<double> angVel_diag_mat (3), angVel_rhs (3), totVel (3+1), angVel_sol(3);

      x = 0; y = 1; z = 2;
      inU = 0; inV = 1; inW = 2;

      for (; cell!=endc; ++cell)
      {
   stokes_fe_values.reinit (cell);
   stokes_fe_values.get_function_values (stokes_solution, stokes_solu);
   stokes_fe_values.get_function_gradients (stokes_solution , stokes_solGrads);

   cell->get_dof_indices (local_dof_indices);

   std::vector<Point<dim> > coor = stokes_fe_values.get_quadrature_points();
   Point<dim> coorCen = cell->center();

   std::pair<unsigned int,double> distant_of_par = 
      distant_from_particles (coorCen, par_rad, a_raTStre);
   if (distant_of_par.second < 0)
   {
       for (unsigned int q=0 ; q < n_q_points ; ++q)
       {
    double r1 = coor[q][x];
    double r2 = coor[q][y];
    double r3 = 0.0; if (dim == 3) r3 = coor[q][z];

    double u1 = stokes_solu[q](inU);
    double u2 = stokes_solu[q](inV);
    double u3 = 0.0; if (dim == 3) u3 = stokes_solu[q](inW);

    for (unsigned int i=0 ; i < n_q_points ; ++i)
    {
        unsigned int in_i = stokes_fe.component_to_system_index (inU,i);

        angVel_diag_mat (z) += stokes_fe_values.shape_value (in_i,q) *
           (r1*r1 + r2*r2)*
           stokes_fe_values.JxW(q);

        angVel_rhs (z) += stokes_fe_values.shape_value (in_i,q) *
      (u1*r2 - u2*r1)*
      stokes_fe_values.JxW(q);

        for (unsigned int j=0; j<n_q_points; ++j)
        {
     unsigned int in_j = stokes_fe.component_to_system_index (inU,j);
     totVel (dim) += stokes_fe_values.shape_value (in_i,q) *
        stokes_fe_values.shape_value (in_j,q) *
        stokes_fe_values.JxW(q);
        }

        totVel(x) += stokes_fe_values.shape_value (in_i,q) *
       stokes_fe_values.JxW(q) *
       u1;

        totVel(y) += stokes_fe_values.shape_value (in_i,q) *
       stokes_fe_values.JxW(q) *
       u2;
     
        if (dim == 3)
        {
     angVel_diag_mat (x) += stokes_fe_values.shape_value (in_i,q) *
       (r2*r2 + r3*r3)*
       stokes_fe_values.JxW(q);

     angVel_diag_mat (y) += stokes_fe_values.shape_value (in_i,q) *
       (r1*r1 + r3*r3)*
       stokes_fe_values.JxW(q);

     angVel_rhs (x) +=  stokes_fe_values.shape_value (in_i,q) *
          (u2*r3 - u3*r2)*
          stokes_fe_values.JxW(q);

     angVel_rhs (y) +=  stokes_fe_values.shape_value (in_i,q) *
          (u3*r1 - u1*r3)*
          stokes_fe_values.JxW(q);

     totVel (z) +=  stokes_fe_values.shape_value (in_i,q) *
      stokes_fe_values.JxW(q) *
      u3;

        }

    }

       }
   }
      }

      if (dim == 2)
   twoDimn_ang_vel[n_pars] = -angVel_rhs(z)/angVel_diag_mat(z);

      if (dim == 3)
      for (unsigned int d=0; d<dim; ++d)
   angVel_sol(d) = -angVel_rhs(d)/angVel_diag_mat(d);

      avrU[n_pars] = totVel(x) / totVel(dim);
      avrV[n_pars] = totVel(y) / totVel(dim);
      if (dim == 3) avrW[n_pars] = totVel(z) / totVel(dim);

      {
   if (dim == 2)
   {
       orient_vector(0) += -twoDimn_ang_vel[0]*orient_vector(1)*dt;
       orient_vector(1) +=  twoDimn_ang_vel[0]*orient_vector(0)*dt;
   }

   if (dim == 3)
   {
       orient_vector(0) += (angVel_sol(1)*orient_vector(2)
      -angVel_sol(2)*orient_vector(1))*dt;

       orient_vector(1) += (angVel_sol(2)*orient_vector(0)
      -angVel_sol(0)*orient_vector(2))*dt;

       orient_vector(2) += (angVel_sol(0)*orient_vector(1)
      -angVel_sol(1)*orient_vector(0))*dt;
   }

   orient_vector /= orient_vector.norm();

   jeff_orb = 0.0;
   tau_nonsp = 0.0;
   alpha_factor = 0.0;
       
   if (dim == 3)
   {
       double dum = 1.0/(1.0 + a_raTStre);
       alpha_factor = std::sqrt(dum);

       double a1 = alpha_factor*orient_vector(2);
       double a2 = alpha_factor*alpha_factor*orient_vector(1)*orient_vector(1) +
     orient_vector(0)*orient_vector(0);

       jeff_orb = std::sqrt(a2)/a1;

       tau_nonsp = atan (180.0*numbers::PI*(1.0/alpha_factor)*
     (orient_vector(0)/orient_vector(1)));

       pcout << a_raTStre << " "
   << dum << " "
   << alpha_factor << " "
   << jeff_orb << " "
   << tau_nonsp << std::endl;
   }
      }

      if (dim == 2)
      out_a  << iii << " "
  << a_raTStre << " "
  << orient_vector << " " 
  << avrU[n_pars] << " "
  << avrV[n_pars] << " "
  << twoDimn_ang_vel[n_pars] 
  << std::endl;
  
      if (dim == 3)
      out_a  << iii << " "
  << num_ele_cover_particle << " " 
  << alpha_factor << " "
        << jeff_orb << " "
  << tau_nonsp << " "
  << a_raTStre << " "
  << orient_vector << " " 
  << avrU[n_pars] << " "
  << avrV[n_pars] << " "
  << avrW[n_pars] << " "
  << angVel_sol 
  << std::endl;

      out_or  << iii << " "
  << orient_vector << std::endl;

  }// n_pars_loop_end or just end } if n_pars = 0;
}


template <int dim>
void ViscoElasticFlow<dim>::compute_stress_system ( 
        unsigned int iii, 
        std::ofstream &out_q1,
        std::ofstream &out_q2)
{
  pcout << "* Compute Stress System..." << std::endl;

  QGauss<dim>  quadrature_formula(3);
  const unsigned int n_q_points = quadrature_formula.size();

  FEValues<dim> stokes_fe_values ( stokes_fe,quadrature_formula,
     UpdateFlags(update_values    |
     update_gradients |
     update_q_points  |
     update_JxW_values));

  FEValues<dim> stress_fe_values ( stress_fe,quadrature_formula,
     UpdateFlags(update_values));
  
  const unsigned int   dofs_per_cell = stokes_fe.dofs_per_cell;
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
      
  std::vector<Vector<double> > stokes_solu (n_q_points , 
         Vector<double>(dim+1));
  std::vector<std::vector<Tensor<1,dim> > > stokes_solGrads ( 
         stokes_fe_values.n_quadrature_points ,
         std::vector<Tensor<1,dim> > (dim+1));
  std::vector<Vector<double> > solu (n_q_points , Vector<double>(number_of_visEl));

  const FEValuesExtractors::Vector velocities (0);
  const FEValuesExtractors::Scalar pressure (dim);

  typename DoFHandler<dim>::active_cell_iterator
    cell = stokes_dof_handler.begin_active(),
    endc = stokes_dof_handler.end(),
    stress_cell =  stress_dof_handler.begin_active();

  double global_effective_viscosity = 0.0;
  std::vector<double> stress_near_particle (3*(dim-1));
  
  for (; cell!=endc; ++cell, ++stress_cell)
  {
    stokes_fe_values.reinit (cell);
    stress_fe_values.reinit (stress_cell);
    
    stokes_fe_values.get_function_values (stokes_solution, stokes_solu);
    stokes_fe_values.get_function_gradients (stokes_solution , stokes_solGrads);
    stress_fe_values.get_function_values (stress_solution, solu);
    
    cell->get_dof_indices (local_dof_indices);
    std::vector<Point<dim> > coor = stokes_fe_values.get_quadrature_points();
    Point<dim> coorCen = cell->center();
    std::pair<unsigned int,double> 
      distant_of_par = 
        distant_from_particles (coorCen, par_rad, a_raTStre);
    std::pair<unsigned int,double> 
      distant_of_par2 = 
        distant_from_particles (coorCen, par_rad + big_ref_rad, a_raTStre);
    
    double nu = 1.0; if (distant_of_par.second < 0) nu = FacPar;
        
    for (unsigned int q=0; q<n_q_points; ++q)
    {
 global_effective_viscosity +=  (
     nu*(stokes_solGrads[q][0][1]
         + 
         stokes_solGrads[q][1][0])
     +
     solu[q][inT12]
     )*stokes_fe_values.JxW(q);
     
 if (distant_of_par2.second < 0.0)
 {
   for (unsigned int i=0; i<3*(dim-1); ++i)
   stress_near_particle [i] += solu[q][i]*stokes_fe_values.JxW(q);
 }
    }
  }
  
  out_q1 << iii << " " << global_effective_viscosity;
  for (unsigned int i=0; i<3*(dim-1); ++i)
    out_q1 << " " << stress_near_particle [i];
  out_q1 << std::endl;
   
}

template <int dim>
void ViscoElasticFlow<dim>::readat ()
{
    pcout << "* Read All Information..." << std::endl;
    
    prm.enter_subsection ("Mesh Information");
      iGmsh =    prm.get_bool   ("Gmesh Input");
      input_mesh_file =  prm.get   ("Input File Name");
      init_level =   prm.get_integer  ("Initial Level");
      max_level =   prm.get_integer  ("Max Level");
      min_level =   prm.get_integer  ("Min Level");
      xmin =    prm.get_double  ("X-axis min");
      xmax =    prm.get_double   ("X-axis max");
      ymin =    prm.get_double   ("Y-axis min");
      ymax =    prm.get_double   ("Y-axis max");
      zmin =    prm.get_double   ("Z-axis min");
      zmax =    prm.get_double   ("Z-axis max");
      small_ref_rad =   prm.get_double   ("Small Radius");
      big_ref_rad =   prm.get_double   ("Big Radius");
      big_asp_ratio =   prm.get_double   ("Big Aspect Ratio");
    prm.leave_subsection ();
    
    prm.enter_subsection ("Particle");
      is_immobile_particle = prm.get_bool ("Immobile Particle");
      factor_for_safeguard = prm.get_double ("Factor of Safe Guard");
      num_pars = prm.get_integer ("No. Particle");
      is_random_particles = prm.get_bool ("Random Particles");
      par_rad = prm.get_double ("Particle Radius");
      FacPar = prm.get_double ("Viscosity Factor");
      a_raTStre = prm.get_double ("Aspect Ratio");
      asp_rat_cylin = prm.get_double ("Aspect ratio for Cylinder");
      ori_x = prm.get_double ("Orientation for X-axis");
      ori_y = prm.get_double ("Orientation for Y-axis");
      ori_z = prm.get_double ("Orientation for Z-axis");
      what_viscous_method = prm.get_integer ("Viscosity Method");
    prm.leave_subsection ();
    
    prm.enter_subsection ("Equation");
      dt = prm.get_double ("Time Interval");
      endCycle = prm.get_integer ("No. of Time Step");
      totVis = prm.get_double ("Total Viscosity");
      tauD = prm.get_double ("Relaxation Time");
      beta = prm.get_double ("Beta");
      shrF = prm.get_double ("Shear Rate");
      osi_amplitude = prm.get_double ("Amplitude");
      osi_ang_freq = prm.get_double ("Angular Frequency");
    prm.leave_subsection ();
    
    prm.enter_subsection ("Restart");
      is_restart = prm.get_bool ("Restart");
      restart_no_timestep = prm.get_integer ("Check Point");
      index_for_restart = prm.get_integer ("Index for VTU");
    prm.leave_subsection ();
    
    prm.enter_subsection ("Problem");
      dimn = prm.get_integer ("Dimension");
      initial_mesh_check = prm.get_bool ("Initial Mesh Check");
      reaDat = prm.get_bool ("Read Data");
      strt_Cycle = prm.get_integer ("Re-Run-i");
      is_recover_triangulation = prm.get_bool ("Recover Trg.");
      is_recover_orientation = prm.get_bool ("Recover Orn.");
      is_periodic = prm.get_bool ("Periodic Bnd.");
      error_stokes = prm.get_double ("Error Stokes");
      error_stress = prm.get_double ("Error Stress");
      type_model = prm.get_integer ("Consti. Model Type");
      model_parameter = prm.get_double ("Model Parameter");
      is_debug_mode = prm.get_bool ("Debug Mode");
      output_fac = prm.get_integer ("Output Period");
      refine_fac = prm.get_integer ("Refine Period");
    prm.leave_subsection ();
    
  inU = 0; inV = 1; inP = 2; inT11 = 0; inT22 = 1; 
  inT12 = 2; inT33 = 3; inT13 = 4; inT23 = 5;
  dx = 0; dy = 1; dz = 2; x = 0; y = 1; z = 2;
  if (dim == 3) inW = 2; inP = 3;

  imposed_vel_at_wall = shrF/std::abs(ymin-ymax);
  
  if (num_pars >0 && par_rad > 0 && FacPar > 1.0) 
    {is_solid_particle = true; particle_generation ();}
    
  orient_vector(0) = ori_x;
  orient_vector(1) = ori_y;
  if (dim == 3) orient_vector (2) = ori_z;
  
  for (unsigned int d=0; d<dim; ++d)
    orient_vector(d) = orient_vector(d)/orient_vector.norm();
    
  local_ref_no = max_level - min_level;
  
  h_min = (xmax-xmin)*(1./std::pow(2.0, max_level));
}


template <int dim>
void ViscoElasticFlow<dim>::particle_generation ()
{
    pcout << "* Particle Generation.. " << std::endl;

    double xlen = std::abs(xmax-xmin);
    double ylen = std::abs(ymax-ymin);
    double zlen = std::abs(zmax-zmin);
    
    if (is_random_particles == 0)
    {
        std::string filename_par = "particle.prm";
        std::ifstream out_par (filename_par.c_str());

 pcout << "* Load Particle Information..." << std::endl;
 
        for (unsigned int n = 0; n < num_pars; ++n)
        {
            double co1, co2, co3, ra;

            Point<dim> raco, image_raco;
            if (dim == 2) out_par >> co1 >> co2;
            if (dim == 3) out_par >> co1 >> co2 >> co3;

            raco[0] = co1;
            raco[1] = co2;
            if (dim == 3) raco[2] = co3;
     
            image_raco = raco;

            if (raco[0] < 0.5*(xmax+xmin) ) image_raco[0] = raco[0] + xlen;
            if (raco[0] > 0.5*(xmax+xmin) ) image_raco[0] = raco[0] - xlen;

            cenPar.push_back(raco);
            image_cenPar.push_back(image_raco);
     twoDimn_ang_vel.push_back (0.0);
     
            pcout <<  cenPar[n] << " | " << image_cenPar[n] << " | " << par_rad << std::endl;
        }
    }


    if (is_random_particles == 1)
    {
        pcout << "* Generate Random Particles.. " << std::endl;

 std::string filename_par = "ini_par";
        std::ofstream out_ini_par (filename_par.c_str());

        std::vector<unsigned int> dum;

        dum.push_back (int(xlen)*10000);
        dum.push_back (int(ylen)*10000);
        dum.push_back (int(zlen)*10000);

        for (unsigned int n = 0 ; n < num_pars ; ++n)
        {
            Point<dim> tmpPar;
            Point<dim> tmpp;
            cenPar.push_back (tmpPar);
            image_cenPar.push_back (tmpPar);

            bool valid_tmp_pars;
            valid_tmp_pars = false;

            unsigned int count = 0;
            do
            {
                bool valid_gen_tmpPar;

                unsigned int t = 0;
                do
                {
                    valid_gen_tmpPar = true;
                    for (unsigned int d = 0; d < dim ; ++d)
                    {
                        unsigned int num0 = rand()%dum[d];
                        double num1 = num0;
                        double leg_max = 0.0;
                        if (d == 0) leg_max = xmax;
                        if (d == 1) leg_max = ymax;
                        if (d == 2) leg_max = zmax;
                        num1 = num1/10000 - leg_max;
                        //num1 = num1/10000;
                        tmpPar[d] = num1;
                    }

                   //double xxmin = xmin+par_rad;
      double xxmin = xmin + par_rad;
      double xxmax = xmax - par_rad;
      double yymin = ymin + par_rad;
      double yymax = ymax - par_rad;
      double half_y_len = 0.5*(ymax+ymin);
      double off_yymin = half_y_len + par_rad;
      double off_yymax = half_y_len - par_rad;


                   if (  (tmpPar[0]  < xxmin || tmpPar[0] > xxmax) ||
     (tmpPar[1]  < yymin || tmpPar[1] > yymax) )
                        valid_gen_tmpPar = false;

                    ++t;
                } while (valid_gen_tmpPar == false);


    cenPar[n] = tmpPar;
    image_cenPar[n] = tmpPar;
    double half_xlen = 0.5*(xmax+xmin);

    if (cenPar[n][0] < half_xlen)  image_cenPar[n][0] = tmpPar[0] + xlen;
    if (cenPar[n][0] > half_xlen)  image_cenPar[n][0] = tmpPar[0] - xlen;

    valid_tmp_pars = true;

    for(unsigned i = 0 ; i < n ; ++i)
    {
                    double dist_cen_cen = 0.0;

                    dist_cen_cen = cenPar[i].distance(tmpPar);

                    if (dist_cen_cen < 2.01*par_rad) valid_tmp_pars = false;

                    dist_cen_cen = image_cenPar[i].distance(tmpPar);

                    if (dist_cen_cen < 2.01*par_rad) valid_tmp_pars = false;
    }

    if (n == 0) valid_tmp_pars = true;

    ++count;

            } while (valid_tmp_pars == false); //do
            pcout <<  cenPar[n] << std::endl;
     out_ini_par <<  cenPar[n] << std::endl;
        }
    }
    for (unsigned n=0; n < num_pars; ++n)
    {
      avrU.push_back (0.0);
      avrV.push_back (0.0);
      avrW.push_back (0.0);
    }
}

template <int dim>
void ViscoElasticFlow<dim>::run ()
{
  srand( (unsigned) time(NULL));
    
  const std::string filename_a = "data_output/compute_particle_dyn_properties.dat";
  const std::string filename_a1 = "data_output/compute_stress1.dat";
  const std::string filename_a2 = "data_output/compute_stress2.dat";
  const std::string filename_or = "data_output/orientation.dat";
  const std::string filename_sa = "data_output/save_time.dat";
  std::ofstream out_a (filename_a.c_str());  
  std::ofstream out_a1 (filename_a1.c_str());  
  std::ofstream out_a2 (filename_a2.c_str());  
  std::ofstream out_or (filename_or.c_str());
  std::ofstream out_sa (filename_sa.c_str());
  
  out_a.precision(8);
  out_or.precision(14);
  
  readat ();
  
  unsigned int index_plotting = std::numeric_limits<unsigned int>::max();
  
  if (is_restart == false)
  {
    create_triangulation (); 
    if (local_ref_no > 0) initial_refined_coarse_mesh ();
    
    if (initial_mesh_check)
    {
      setup_dofs(false);
      if (local_ref_no > 0) 
      for (unsigned int i=0; i<local_ref_no-1; ++i) refine_mesh (false);
//       viscosity_distribution ();  
//       plotting_solution (0);
//       return;
    }
      
//     if (num_pars > 0 && is_immobile_particle == false)
//     compute_particle_dyn_properties (
//      timestep_number, 
//      out_a, 
//      out_or);
  
    index_plotting = 0;
  }
  else if (is_restart)
  {
    timestep_number = restart_no_timestep + 1;
    index_plotting = index_for_restart + 1;
    resume_snapshot ();
  }

  setup_dofs(true);
   
  do
  {
    pcout << "" << std::endl;
    pcout <<"# No. = " << timestep_number << std::endl;
        
    imposed_vel_at_wall = shrF;
    if (osi_ang_freq > 0 && osi_amplitude > 0)
      imposed_vel_at_wall = osi_ang_freq*osi_amplitude*
    std::cos(osi_ang_freq*timestep_number*dt);
     
    if (timestep_number > 0 && tauD > 1e-11)
    {
      assemble_peri_DGConvect (); 
      assemble_RHSVisEls ();
      for (unsigned int j = 0 ; j<number_of_visEl ; ++j) solve (1, j);
    }
    
    viscosity_distribution ();
    build_stokes_preconditioner ();
    assemble_stokes_system ();
    solve (0,0);

//    if (Utilities::Trilinos::get_this_mpi_process(trilinos_communicator)==0)
 {       
      if (num_pars > 0 && is_immobile_particle == false)
      compute_particle_dyn_properties( timestep_number + 1, 
       out_a, 
       out_or);
  
      compute_stress_system( timestep_number + 1, 
     out_a1, 
     out_a2);
  
      bool allow_print = false;
      bool allow_refinement = false;
    
      if (timestep_number%output_fac == 0)
      {
        plotting_solution (index_plotting);
        save_snapshot (index_plotting);
        ++index_plotting;
      }
 }
    pcout << "# num_ele_cover_particle = "
  << num_ele_cover_particle/double(num_pars)
  << " | " 
  << (num_ele_cover_particle/double(num_pars))*
      std::pow(h_min, double(dim))
  << std::endl;

    if( timestep_number%refine_fac == 0 &&
        endCycle > 1 &&
        is_immobile_particle == false)
      refine_mesh (true);
    ++ timestep_number;

  } while (timestep_number < endCycle);
}

int main (int argc, char *argv[])
{
  try
    {
      deallog.depth_console (0);

      Utilities::System::MPI_InitFinalize mpi_initialization(argc, argv);

 ParameterHandler  prm;
 ParameterReader   param(prm);
 param.read_parameters("input.prm");

 prm.enter_subsection ("Problem");
  unsigned int dimn = prm.get_integer ("Dimension");
 prm.leave_subsection ();
     
 switch (dimn)
 {
   case 2 : 
   {
     ViscoElasticFlow<2>  ysh (prm);
     ysh.run ();
   }
     break;
   case 3 :
   {
     ViscoElasticFlow<3>  ysh (prm);
     ysh.run ();
   }
     break; 
 }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
