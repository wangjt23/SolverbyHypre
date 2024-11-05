    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include "HYPRE_krylov.h"
    #include "HYPRE.h"
    #include "HYPRE_parcsr_ls.h"
    #include "_hypre_utilities.h"
    #include "_hypre_parcsr_mv.h"
    // matrix read from a single file (CSR format
    HYPRE_Int BuildParFromOneFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                HYPRE_Int num_functions, HYPRE_ParCSRMatrix *A_ptr );
    // rhs read from a single file (CSR format)
    HYPRE_Int BuildRhsParFromOneFile (HYPRE_Int argc, char *argv [], HYPRE_Int arg_index,
                                    HYPRE_ParCSRMatrix A, HYPRE_ParVector *b_ptr );


    int main (int argc, char *argv[])
    {
    int myid, num_procs;
    int first_local_row, last_local_row, first_local_col, last_local_col;

    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;
    HYPRE_Solver solver, precond;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* Initialize HYPRE */
    HYPRE_Initialize();

    /* Default problem parameters */
    int solver_id = 0;
    int build_matrix_arg_index=-1;
    int build_rhs_arg_index   =-1;
    double solver_tol         = 1.e-6;
    int solver_max_iter_nums  = 100;
    int saveSol_index = -1;
    // AMG default prameters
    int amg_print_level       = 1;
    int amg_num_sweeps        = 1;
    int amg_corase_type       = 8; // PMIS
    int amg_prolongation_type = 0;
    int amg_max_coarsen_size  = 9;
    int amg_max_levels        = 25;
    int amg_precond_iters     = 1;
    double amg_strong_threshold  = 0.25;
    int amg_cycle_type        = 1; //1: V-cycle; 2: W-cycle 
    // PCG default parameters

    // Gmres default parameters
    int gmres_max_dim_size    = 10;
    int gmres_print_info      = 2; // 2: print;  0: not print 
    // ILU
    int ilu_type              = 10;
    int ilu_lfil              = 0;
    int ilu_reordering        = 0;
    int ilu_max_row_nnz       = 1000;
    double ilu_droptol       = 1.e-2;

    /* Parse command line */
    {
        int arg_index = 0;
        int print_usage = 0;
        
        while (arg_index < argc)
        {
            if ( strcmp(argv[arg_index], "-solver") == 0 )
            {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
            }
            else if (strcmp(argv[arg_index], "-maxIters") == 0)
            {
                arg_index++;
                solver_max_iter_nums = atoi(argv[arg_index++]);
            }
            else if (strcmp(argv[arg_index], "-tol") == 0)
            {
                arg_index++;
                solver_tol = atof(argv[arg_index++]);
            }
            else if ( strcmp(argv[arg_index], "-help") == 0 )
            {
            print_usage = 1;
            break;
            }
            else if (strcmp(argv[arg_index], "-matPath") == 0 )
            {
            arg_index ++;
            build_matrix_arg_index = arg_index++;
            }
            else if (strcmp(argv[arg_index], "-rhsPath") == 0 )
            {
            arg_index ++;
            build_rhs_arg_index = arg_index++;
            }
            else if (strcmp(argv[arg_index], "-SolPath") == 0 )
            {
            arg_index ++;
            saveSol_index = arg_index++;
            }
            else
            {
            arg_index++;
            }
        }

        if ((print_usage) && (myid == 0))
        {
            printf("\n");
            printf("Usage: %s [<options>]\n", argv[0]);
            printf("\n");
            printf("  -solver <ID>          : solver ID\n");
            printf("                        0  - AMG (default) \n");
            printf("                        1  - PCG-AMG\n");
            printf("                        3  - GMRES-AMG\n");
            printf("                        80 - ILU\n");
            printf("  -matPath <FILEDIR>    : csr mat file path\n");
            printf("  -rhsPath <FILEDIR>    : rhs vec file path\n");
            printf("  -maxIters <n>         : set solver's max iterations (default: 100)\n");
            printf("  -tol <value>          : set solver's tolerance (default: 1e-6)\n");
            printf("  -SolPath <FILEDIR>    : whether save solution in one file(given the file dir)(default: not)\n");
            // printf("  -print_system       : print the matrix and rhs\n");
            printf("\n");
        }

        if (print_usage)
        {
            MPI_Finalize();
            return (0);
        }
    }
    // debug 
    if (myid ==0){
        if (build_matrix_arg_index == -1)
        {
            fprintf(stderr, "Error: MatPath has not been properly assigned!\n");
            exit(EXIT_FAILURE);
        }
        if (build_rhs_arg_index == -1)
        {
            fprintf(stderr, "Error: rhsPath has not been properly assigned!\n");
            exit(EXIT_FAILURE);
        } 
    }
 
    BuildParFromOneFile(argc, argv, build_matrix_arg_index, 1,&parcsr_A);
    BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, parcsr_A, &par_b);
    HYPRE_ParCSRMatrixGetLocalRange(parcsr_A,
                                    &first_local_row, &last_local_row,
                                    &first_local_col, &last_local_col );
    /* initial guess */
    HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);
    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(x, (void **) &par_x );
    // HYPRE_IJVectorDestroy(x);

    /* Choose a solver and solve the system */

    /* AMG */
    if (solver_id == 0)
    {
        int num_iterations;
        double final_res_norm;
        double max_setup_time,max_solve_time;
        /* Create solver */
        double start_setup_time = MPI_Wtime();
        HYPRE_BoomerAMGCreate(&solver);
        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_BoomerAMGSetPrintLevel(solver, amg_print_level);  /* print solve info + parameters */
        // HYPRE_BoomerAMGSetOldDefault(solver); /* Falgout coarsening with modified classical interpolaiton */
        // HYPRE_BoomerAMGSetRelaxType(solver, 0);   /* G-S/Jacobi hybrid relaxation */
        // HYPRE_BoomerAMGSetRelaxOrder(solver, 0);   /* uses C/F relaxation */
        HYPRE_BoomerAMGSetNumSweeps(solver, amg_num_sweeps);   /* Sweeeps on each level */
        HYPRE_BoomerAMGSetCoarsenType(solver, amg_corase_type);/* PMIS*/
        HYPRE_BoomerAMGSetMaxCoarseSize(solver, amg_max_coarsen_size);
        HYPRE_BoomerAMGSetMaxLevels(solver, amg_max_levels);  /* maximum number of levels */
        HYPRE_BoomerAMGSetTol(solver, solver_tol);      /* conv. tolerance */
        HYPRE_BoomerAMGSetMaxIter(solver, solver_max_iter_nums); 
        HYPRE_BoomerAMGSetStrongThreshold(solver, amg_strong_threshold);
        HYPRE_BoomerAMGSetCycleType(solver, amg_cycle_type);
        /* Now setup and solve! */
        HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
        double end_setup_time = MPI_Wtime();
        double setup_time = end_setup_time - start_setup_time;
        MPI_Reduce(&setup_time,&max_setup_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

        /*Solve*/
        double start_solve_time = MPI_Wtime();
        HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
        double end_solve_time = MPI_Wtime();
        double solve_time = end_solve_time - start_solve_time;
        MPI_Reduce(&solve_time,&max_solve_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        /* Run info - needed logging turned on */
        HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
        HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
        if (myid == 0)
        {
            printf("\n");
            printf("AMG Iterations = %d\n", num_iterations);
            printf("Relative Residual Norm = %e\n", final_res_norm);
            printf("Setup time: %.4f s\n", max_setup_time);
            printf("Solve time: %.4f s\n", max_solve_time);
            // printf("Total time: %.4f s\n",max_setup_time+max_solve_time);
            printf("\n");
        }

        /* Destroy solver */
        HYPRE_BoomerAMGDestroy(solver);
    }
    /* PCG with AMG preconditioner */
    else if (solver_id == 1)
    {
        int num_iterations;
        double final_res_norm;
        double max_setup_time,max_solve_time;
        /* Create solver */
        double start_setup_time = MPI_Wtime();
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_PCGSetMaxIter(solver, solver_max_iter_nums); /* max iterations */
        HYPRE_PCGSetTol(solver, solver_tol); /* conv. tolerance */
        HYPRE_PCGSetTwoNorm(solver, 1); /* use the two norm as the stopping criteria */
        HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
        HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */
        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);
        HYPRE_BoomerAMGSetPrintLevel(precond, amg_print_level); /* print amg solution info */
        HYPRE_BoomerAMGSetCoarsenType(precond, amg_corase_type);/*PMIS*/
        HYPRE_BoomerAMGSetInterpType( precond, amg_prolongation_type);
        HYPRE_BoomerAMGSetMaxLevels(precond, amg_max_levels);
        HYPRE_BoomerAMGSetPMaxElmts( precond,0);/*Turn of truncation*/
        HYPRE_BoomerAMGSetRelaxType(precond, 0); /* Sym G.S./Jacobi hybrid */
        HYPRE_BoomerAMGSetNumSweeps(precond, amg_num_sweeps);
        HYPRE_BoomerAMGSetTol(precond, 0); /* conv. tolerance zero */
        HYPRE_BoomerAMGSetMaxIter(precond, amg_precond_iters); /* do only one iteration! */
        /* Set the PCG preconditioner */
        HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                            (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
        /* Now setup and solve! */
        HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
        double end_setup_time = MPI_Wtime();
        double setup_time = end_setup_time - start_setup_time;
        MPI_Reduce(&setup_time,&max_setup_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        /* Solve*/
        double start_solve_time = MPI_Wtime();
        HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
        double end_solve_time = MPI_Wtime();
        double solve_time = end_solve_time - start_solve_time;
        MPI_Reduce(&solve_time,&max_solve_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        /* Run info - needed logging turned on */
        HYPRE_PCGGetNumIterations(solver, &num_iterations);
        HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
        if (myid == 0)
        {
            printf("\n");
            printf("PCG&AMG Iterations = %d\n", num_iterations);
            printf("Final Relative Residual Norm = %e\n", final_res_norm);
            printf("Setup time: %.4f s\n", max_setup_time);
            printf("Solve time: %.4f s\n", max_solve_time);
            printf("\n");
        }

        /* Destroy solver and preconditioner */
        HYPRE_ParCSRPCGDestroy(solver);
        HYPRE_BoomerAMGDestroy(precond);
    }
    /* AMG + Gmres*/
    else if (solver_id == 3)
    {
        double final_res_norm;
        int num_iterations;
        double max_setup_time,max_solve_time;

        /*Gmres Setup*/
        double start_setup_time = MPI_Wtime();
        HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);
        HYPRE_GMRESSetKDim(solver, gmres_max_dim_size);
        HYPRE_GMRESSetMaxIter(solver, solver_max_iter_nums);
        HYPRE_GMRESSetTol(solver, solver_tol);
        // HYPRE_GMRESSetAbsoluteTol(solver, atol);
        HYPRE_GMRESSetLogging(solver, 1);
        HYPRE_GMRESSetPrintLevel(solver, gmres_print_info);
        // HYPRE_GMRESSetRelChange(solver, 1);

        /* AMG Setup*/
        HYPRE_BoomerAMGCreate(&precond);
        HYPRE_BoomerAMGSetPrintLevel(precond, amg_print_level); /* print amg solution info */
        HYPRE_BoomerAMGSetCoarsenType(precond, amg_corase_type);/*PMIS*/
        HYPRE_BoomerAMGSetInterpType( precond, amg_prolongation_type);
        HYPRE_BoomerAMGSetMaxCoarseSize(precond, amg_max_coarsen_size);
        HYPRE_BoomerAMGSetMaxLevels(precond, amg_max_levels);
        HYPRE_BoomerAMGSetPMaxElmts( precond,0);/*Turn of truncation*/
        HYPRE_BoomerAMGSetStrongThreshold(precond, amg_strong_threshold);
        HYPRE_BoomerAMGSetCycleType(precond, amg_cycle_type);
        // HYPRE_BoomerAMGSetRelaxType(precond, 0); /* Sym G.S./Jacobi hybrid */
        HYPRE_BoomerAMGSetNumSweeps(precond, amg_num_sweeps);
        HYPRE_BoomerAMGSetTol(precond, 0); /* conv. tolerance zero */
        HYPRE_BoomerAMGSetMaxIter(precond, amg_precond_iters); /* do only one iteration! */

        HYPRE_GMRESSetPrecond(solver,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                precond);
        /*Set up all */
        HYPRE_GMRESSetup(solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)par_b, (HYPRE_Vector)par_x); 

        double end_setup_time = MPI_Wtime();
        double setup_time = end_setup_time - start_setup_time;
        MPI_Reduce(&setup_time,&max_setup_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        
        /* Solve Phase*/
        double start_solve_time = MPI_Wtime();
        HYPRE_GMRESSolve(solver, (HYPRE_Matrix)parcsr_A, (HYPRE_Vector)par_b, (HYPRE_Vector)par_x); 
        double end_solve_time = MPI_Wtime();
        double solve_time = end_solve_time - start_solve_time;
        MPI_Reduce(&solve_time,&max_solve_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

        HYPRE_GMRESGetNumIterations(solver, &num_iterations);
        HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
        if (myid == 0)
        {
            printf("\n");
            printf("Gmres&AMG Iterations = %d\n", num_iterations);
            printf("Final Relative Residual Norm = %e\n", final_res_norm);
            printf("Setup time: %.4f s\n", max_setup_time);
            printf("Solve time: %.4f s\n", max_solve_time);
            printf("\n");
        }
        HYPRE_BoomerAMGDestroy(precond);
        HYPRE_ParCSRGMRESDestroy(solver);
      
    }
    else if (solver_id == 80)
    {
        int num_iterations;
        double final_res_norm;
        double max_setup_time,max_solve_time;

        double start_setup_time = MPI_Wtime();
        HYPRE_ILUCreate(&solver);

        /* set ilu type */
        HYPRE_ILUSetType(solver, ilu_type);
        /* set level of fill */
        HYPRE_ILUSetLevelOfFill(solver, ilu_lfil);
        /* set local reordering type */
        HYPRE_ILUSetLocalReordering(solver, ilu_reordering);
        /* set print level */
        HYPRE_ILUSetPrintLevel(solver, 3);
        /* set max iterations */
        HYPRE_ILUSetMaxIter(solver, solver_max_iter_nums);
        /* set max number of nonzeros per row */
        HYPRE_ILUSetMaxNnzPerRow(solver, ilu_max_row_nnz);
        /* set the droptol */
        HYPRE_ILUSetDropThreshold(solver, ilu_droptol);
        HYPRE_ILUSetTol(solver, solver_tol);
        /* set max iterations for Schur system solve */
        // HYPRE_ILUSetSchurMaxIter(solver, ilu_schur_max_iter);
        // HYPRE_ILUSetIterativeSetupType(solver, ilu_iter_setup_type);
        // HYPRE_ILUSetIterativeSetupOption(solver, ilu_iter_setup_option);
        // HYPRE_ILUSetIterativeSetupMaxIter(solver, ilu_iter_setup_max_iter);
        // HYPRE_ILUSetIterativeSetupTolerance(solver, ilu_iter_setup_tolerance);
        /* setting for NSH */
        // if (ilu_type == 20 || ilu_type == 21)
        // {
        //     HYPRE_ILUSetNSHDropThreshold( solver, ilu_nsh_droptol);
        // }


        /* setup hypre_ILU solver */
        HYPRE_ILUSetup(solver, parcsr_A, par_b, par_x);
        double end_setup_time = MPI_Wtime();
        double setup_time = end_setup_time - start_setup_time;
        MPI_Reduce(&setup_time,&max_setup_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        /* hypre_ILU solve */
        double start_solve_time = MPI_Wtime();
        HYPRE_ILUSolve(solver, parcsr_A, par_b, par_x);
        double end_solve_time = MPI_Wtime();
        double solve_time = end_solve_time - start_solve_time;
        MPI_Reduce(&solve_time,&max_solve_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

        HYPRE_ILUGetNumIterations(solver, &num_iterations);
        HYPRE_ILUGetFinalRelativeResidualNorm(solver, &final_res_norm);

        if (myid == 0)
        {
            hypre_printf("\n");
            hypre_printf("ILU Iterations = %d\n", num_iterations);
            hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
            printf("Setup time: %.4f s\n", max_setup_time);
            printf("Solve time: %.4f s\n", max_solve_time);
            hypre_printf("\n");
        }

        /* free memory */
        HYPRE_ILUDestroy(solver);
    }
    else
    {
        if (myid == 0) { printf("Invalid solver id specified.\n"); }
    }

    /* Save solution*/
    if(saveSol_index>=0) 
    {
        hypre_Vector* sol_x;
        sol_x= hypre_ParVectorToVectorAll((hypre_ParVector *)par_x);
        if(myid==0) HYPRE_VectorPrint((HYPRE_Vector)sol_x, argv[saveSol_index]);
        if(myid==0) printf("Solution have saved in %s;\n",argv[saveSol_index]);
        // HYPRE_ParVectorPrint(par_x,argv[saveSol_index]);
    }
    hypre_ParCSRMatrixPrintIJ(parcsr_A, 0, 0, "myij.out.A");
    

    /* Clean up */
    HYPRE_ParCSRMatrixDestroy(parcsr_A);
    hypre_ParVectorDestroy(par_b);

    /* Finalize HYPRE */
    HYPRE_Finalize();

    /* Finalize MPI*/
    MPI_Finalize();

    return (0);
    }



    HYPRE_Int BuildParFromOneFile(  HYPRE_Int           argc,
                                char               *argv[],
                                HYPRE_Int           arg_index,
                                HYPRE_Int           num_functions,
                                HYPRE_ParCSRMatrix *A_ptr     )
    {
    char               *filename;

    HYPRE_CSRMatrix  A_CSR = NULL;
    HYPRE_BigInt       *row_part = NULL;
    HYPRE_BigInt       *col_part = NULL;

    HYPRE_Int          myid, numprocs;
    HYPRE_Int          i, rest, size, num_nodes, num_dofs;

    /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

    MPI_Comm_rank(MPI_COMM_WORLD, &myid );
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs );

    /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

    if (arg_index < argc)
    {
        filename = argv[arg_index];
    }
    else
    {
        hypre_printf("Error: No filename specified \n");
        exit(1);
    }

    /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

    if (myid == 0)
    {
        hypre_printf("  FromFile: %s\n", filename);

        /*-----------------------------------------------------------
        * Generate the matrix
        *-----------------------------------------------------------*/

        A_CSR = HYPRE_CSRMatrixRead(filename);
    }

    if (num_functions !=1){
        fprintf(stderr, "Do not support num functions != 1! \n");
    }

    if (myid == 0 && num_functions==1)
    {
        HYPRE_CSRMatrixGetNumRows(A_CSR, &num_dofs);
        num_nodes = num_dofs;
        row_part = hypre_CTAlloc(HYPRE_BigInt,  numprocs + 1, HYPRE_MEMORY_HOST);

        row_part[0] = 0;
        size = num_nodes / numprocs;
        rest = num_nodes - size * numprocs;
        for (i = 0; i < rest; i++)
        {
        row_part[i + 1] = row_part[i] + size + 1;
        }
        for (i = rest; i < numprocs; i++)
        {
        row_part[i + 1] = row_part[i] + size;
        }

        col_part = row_part;
    }
    HYPRE_CSRMatrixToParCSRMatrix(MPI_COMM_WORLD, A_CSR, row_part, col_part, A_ptr);
    // reorder parcsr 
    hypre_ParCSRMatrixReorder(*A_ptr);
    if (myid == 0)
    {
        HYPRE_CSRMatrixDestroy(A_CSR);
    }

    return (0);
    }

    HYPRE_Int BuildRhsParFromOneFile(   HYPRE_Int            argc,
                                    char                *argv[],
                                    HYPRE_Int            arg_index,
                                    HYPRE_ParCSRMatrix   parcsr_A,
                                    HYPRE_ParVector     *b_ptr     )
    {
    char           *filename;
    HYPRE_Int       myid;
    int   *partitioning;
    HYPRE_ParVector b;
    HYPRE_Vector    b_CSR = NULL;

    /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

    hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
    partitioning = hypre_ParCSRMatrixRowStarts(parcsr_A);

    /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

    if (arg_index < argc)
    {
        filename = argv[arg_index];
    }
    else
    {
        hypre_printf("Error: No filename specified \n");
        exit(1);
    }

    /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

    if (myid == 0)
    {
        hypre_printf("  Rhs FromFile: %s\n", filename);

        /*-----------------------------------------------------------
        * Generate the matrix
        *-----------------------------------------------------------*/

        b_CSR = HYPRE_VectorRead(filename);
    }
    HYPRE_VectorToParVector(hypre_MPI_COMM_WORLD, b_CSR, partitioning, &b);
    *b_ptr = b;

    HYPRE_VectorDestroy(b_CSR);

    return (0);
    }
