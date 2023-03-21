import argparse
import os
import sys
from train import get_train_test_files, train_and_test_model, write_results_file, combine_fold_results

# Check if crossdocked files have been downloaded 

# Create new types file by removing affinity data and readding chosen data 





if __name__ == '__main__':
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m','--model',type=str,required=True,help="Model template. Must use TRAINFILE and TESTFILE", default='default2018.model')
    parser.add_argument('-p','--prefix',type=str,required=True,help="Prefix for training/test files: <prefix>[train|test][num].types")
    parser.add_argument('-d','--data_root',type=str,required=False,help="Root folder for relative paths in train/test files",default='')
    parser.add_argument('-n','--foldnums',type=str,required=False,help="Fold numbers to run, default is to determine using glob",default=None)
    parser.add_argument('-a','--allfolds',action='store_true',required=False,help="Train and test file with all data folds, <prefix>.types",default=False)
    parser.add_argument('-i','--iterations',type=int,required=False,help="Number of iterations to run,default 250,000",default=250000)
    parser.add_argument('-s','--seed',type=int,help="Random seed, default 42",default=42)
    parser.add_argument('-t','--test_interval',type=int,help="How frequently to test (iterations), default 1000",default=1000)
    parser.add_argument('-o','--outprefix',type=str,help="Prefix for output files, default <model>.<pid>",default='')
    parser.add_argument('-g','--gpu',type=int,help='Specify GPU to run on',default=-1)
    parser.add_argument('-c','--cont',type=int,help='Continue a previous simulation from the provided iteration (snapshot must exist)',default=0)
    parser.add_argument('-k','--keep',action='store_true',default=False,help="Don't delete prototxt files")
    parser.add_argument('-r', '--reduced', action='store_true',default=False,help="Use a reduced file for model evaluation if exists(<prefix>[reducedtrain|reducedtest][num].types). Incompatible with --percent_reduced")
    parser.add_argument('--percent_reduced',type=float,default=0,help='Create a reduced set on the fly based on types file, using the given percentage: to use 10 percent pass 10. Range (0,100). Incompatible with --reduced')
    parser.add_argument('--avg_rotations', action='store_true',default=False, help="Use the average of the testfile's 24 rotations in its evaluation results")
    parser.add_argument('--checkpoint', action='store_true',default=False,help="Enable automatic checkpointing")
    #parser.add_argument('-v,--verbose',action='store_true',default=False,help='Verbose output')
    parser.add_argument('--keep_best',action='store_true',default=False,help='Store snapshots everytime test AUC improves')
    parser.add_argument('--dynamic',action='store_true',default=True,help='Attempt to adjust the base_lr in response to training progress, default True')
    parser.add_argument('--cyclic',action='store_true',default=False,help='Vary base_lr in range of values: 0.015 to 0.001')
    parser.add_argument('--solver',type=str,help="Solver type. Default is SGD",default='SGD')
    parser.add_argument('--lr_policy',type=str,help="Learning policy to use. Default is fixed.",default='fixed')
    parser.add_argument('--step_reduce',type=float,help="Reduce the learning rate by this factor with dynamic stepping, default 0.1",default='0.1')
    parser.add_argument('--step_end',type=float,help='Terminate training if learning rate gets below this amount',default=0)
    parser.add_argument('--step_end_cnt',type=float,help='Terminate training after this many lr reductions',default=3)
    parser.add_argument('--step_when',type=int,help="Perform a dynamic step (reduce base_lr) when training has not improved after this many test iterations, default 5",default=5)
    parser.add_argument('--base_lr',type=float,help='Initial learning rate, default 0.01',default=0.01)
    parser.add_argument('--momentum',type=float,help="Momentum parameters, default 0.9",default=0.9)
    parser.add_argument('--weight_decay',type=float,help="Weight decay, default 0.001",default=0.001)
    parser.add_argument('--gamma',type=float,help="Gamma, default 0.001",default=0.001)
    parser.add_argument('--power',type=float,help="Power, default 1",default=1)
    parser.add_argument('--weights',type=str,help="Set of weights to initialize the model with")
    parser.add_argument('-p2','--prefix2',type=str,required=False,help="Second prefix for training/test files for combined training: <prefix>[train|test][num].types")
    parser.add_argument('-d2','--data_root2',type=str,required=False,help="Root folder for relative paths in second train/test files for combined training",default='')
    parser.add_argument('--data_ratio',type=float,required=False,help="Ratio to combine training data from 2 sources",default=None)
    parser.add_argument('--test_only',action='store_true',default=False,help="Don't train, just evaluate test nets once")
    parser.add_argument('--clip_gradients',type=float,default=10.0,help="Clip gradients threshold (default 10)")
    parser.add_argument('--skip_full',action='store_true',default=False,help='Use reduced testset on final evaluation, requires passing --reduced')
    parser.add_argument('--display_iter',type=int,default=0,help='Print out network outputs every so many iterations')
    parser.add_argument('--update_ratio',type=float,default=0.001,help="Improvements during training need to be better than this ratio. IE (best-current)/best > update_ratio. Defaults to 0.001")
    args = parser.parse_args()

    argdict = vars(args)
    line = ''
    for (name,val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' %(name,val)

    (args,cmdline) = (args,line)

    #identify all train/test pairs
    try:
        train_test_files = get_train_test_files(args.prefix, args.foldnums, args.allfolds, args.reduced, args.prefix2, args.percent_reduced)
    except OSError as e:
        print("error: %s" % e)
        sys.exit(1)

    if len(train_test_files) == 0:
        print("error: missing train/test files")
        sys.exit(1)

    if args.percent_reduced < 0 or args.percent_reduced >= 100:
        print("error: percent_reduced must be greater than 0 and less than 100")
        sys.exit(1)

    if args.reduced and args.percent_reduced:
        print("error: can't use reduced and percent_reduced together")
        sys.exit(1)

    if args.skip_full and (not args.reduced and not args.percent_reduced):
        print("error: --skip_full requires --reduced OR --percent_reduced. Neither was not passed")
        sys.exit(1)

    if not (0<args.update_ratio<1):
        print("error: --update_ratio is out of possible values: (0,1)")
        sys.exit(1)

    if args.update_ratio > 0.01:
        print("warning: --update_ratio > 0.01, this may cause earlier termination that desired.")
    
    for i in train_test_files:
        for key in sorted(train_test_files[i], key=len):
            print(str(i).rjust(3), key.rjust(14), train_test_files[i][key])

    outprefix = args.outprefix
    if outprefix == '':
        outprefix = '%s.%d' % (os.path.splitext(os.path.basename(args.model))[0],os.getpid())
        args.outprefix = outprefix

    test_aucs, train_aucs = [], []
    test_rmsds, train_rmsds = [], []
    test_y_true, train_y_true = [], []
    test_y_score, train_y_score = [], []
    test_y_aff, train_y_aff = [], []
    test_y_predaff, train_y_predaff = [], []
    test_rmsd_rmses,train_rmsd_rmses = [], []
    test_rmsd_pred, train_rmsd_pred = [], []
    test_rmsd_true, train_rmsd_true = [], []
    test2_aucs, train2_aucs = [], []
    test2_rmsds, train2_rmsds = [], []
    test2_y_true, train2_y_true = [], []
    test2_y_score, train2_y_score = [], []
    test2_y_aff, train2_y_aff = [], []
    test2_y_predaff, train2_y_predaff = [], []

    checkfold = -1
    if args.checkpoint:
        #check for existence of checkpoint
        cmdcheckname = '%s.cmdline.CHECKPOINT'%outprefix
        if os.path.exists(cmdcheckname):
            #validate this is the same
            #figure out where we were
            oldline = open(cmdcheckname).read()
            if oldline != cmdline:
                print(oldline)
                print("Previous commandline from checkpoint does not match current.  Cannot restore checkpoint.")
                sys.exit(1)
        
        outcheck = open(cmdcheckname,'w')
        outcheck.write(cmdline)
        outcheck.close()        
        
    #train each pair
    numfolds = 0
    for i in train_test_files:

        outname = '%s.%s' % (outprefix, i)        
        cont = args.cont
                
        results = train_and_test_model(args, train_test_files[i], outname, cont)

        if args.prefix2:
            test, train, test2, train2 = results
        else:
            test, train = results

        #write out the final predictions for test and train sets
        if test.aucs:
            write_results_file('%s.auc.finaltest' % outname, test.y_true, test.y_score, footer='AUC %f\n' % test.aucs[-1])
            write_results_file('%s.auc.finaltrain' % outname, train.y_true, train.y_score, footer='AUC %f\n' % train.aucs[-1])

        if test.rmsds:
            write_results_file('%s.rmsd.finaltest' % outname, test.y_aff, test.y_predaff, footer='RMSD %f\n' % test.rmsds[-1])
            write_results_file('%s.rmsd.finaltrain' % outname, train.y_aff, train.y_predaff, footer='RMSD %f\n' % train.rmsds[-1])

        if test.rmsd_rmses:
            write_results_file('%s.rmsd_rmse.finaltest' % outname, test.rmsd_true, test.rmsd_pred, footer='RMSE %f\n' % test.rmsd_rmses[-1])
            write_results_file('%s.rmsd_rmse.finaltrain' % outname, train.rmsd_true, train.rmsd_pred, footer='RMSE %f\n' % train.rmsd_rmses[-1])

        if args.prefix2:
            if test2.aucs:
                write_results_file('%s.auc.finaltest2' % outname, test2.y_true, test2.y_score, footer='AUC %f\n' % test2.aucs[-1])
                write_results_file('%s.auc.finaltrain2' % outname, train2.y_true2, train2.y_score, footer='AUC %f\n' % train2.aucs[-1])

            if test2.rmsds:
                write_results_file('%s.rmsd.finaltest2' % outname, test2.y_aff, test2.y_predaff, footer='RMSD %f\n' % test2.rmsds[-1])
                write_results_file('%s.rmsd.finaltrain2' % outname, train2.y_aff, train2.y_predaff, footer='RMSD %f\n' % train2.rmsds[-1])

        if i == 'all':
            continue
        numfolds += 1

        #aggregate results from different crossval folds
        if test.aucs:
            test_aucs.append(test.aucs)
            train_aucs.append(train.aucs)
            test_y_true.extend(test.y_true)
            test_y_score.extend(test.y_score)
            train_y_true.extend(train.y_true)
            train_y_score.extend(train.y_score)

        if test.rmsds:
            test_rmsds.append(test.rmsds)
            train_rmsds.append(train.rmsds)
            test_y_aff.extend(test.y_aff)
            test_y_predaff.extend(test.y_predaff)
            train_y_aff.extend(train.y_aff)
            train_y_predaff.extend(train.y_predaff)
            
        if test.rmsd_rmses:
            test_rmsd_rmses.append(test.rmsd_rmses)
            train_rmsd_rmses.append(train.rmsd_rmses)
            test_rmsd_true.extend(test.rmsd_true)
            test_rmsd_pred.extend(test.rmsd_pred)
            train_rmsd_true.extend(train.rmsd_true)
            train_rmsd_pred.extend(train.rmsd_pred)            

        if args.prefix2:
            if test2.aucs:
                test2_aucs.append(test2.aucs)
                train2_aucs.append(train2.aucs)
                test2_y_true.extend(test2.y_true)
                test2_y_score.extend(test2.y_score)
                train2_y_true.extend(train2.y_true)
                train2_y_score.extend(train2.y_score)

            if test2.rmsds:
                test2_rmsds.append(test2.rmsds)
                train2_rmsds.append(train2.rmsds)
                test2_y_aff.extend(test2.y_aff)
                test2_y_predaff.extend(test2.y_predaff)
                train2_y_aff.extend(train2.y_aff)
                train2_y_predaff.extend(train2.y_predaff)

    #only combine fold results if we have multiple folds
    if numfolds > 1:

        if any(test_aucs):
            combine_fold_results(test_aucs, train_aucs, test_y_true, test_y_score, train_y_true, train_y_score,
                                 outprefix, args.test_interval, 'pose', second_data_source=False)

        if any(test_rmsds):
            combine_fold_results(test_rmsds, train_rmsds, test_y_aff, test_y_predaff, train_y_aff, train_y_predaff,
                                 outprefix, args.test_interval, 'affinity', second_data_source=False,
                                 filter_actives_test=test_y_true, filter_actives_train=train_y_true)

        if any(test_rmsd_rmses):
            combine_fold_results(test_rmsd_rmses, train_rmsd_rmses, test_rmsd_true, test_rmsd_pred, train_rmsd_true, train_rmsd_pred,
                                 outprefix, args.test_interval, 'rmsd', second_data_source=False)
                                 
                                 
        if any(test2_aucs):
            combine_fold_results(test2_aucs, train2_aucs, test2_y_true, test2_y_score, train2_y_true, train2_y_score,
                                 outprefix, args.test_interval, 'pose', second_data_source=True)

        if any(test2_rmsds):
            combine_fold_results(test2_rmsds, train2_rmsds, test2_y_aff, test2_y_predaff, train2_y_aff, train2_y_predaff,
                                 outprefix, args.test_interval, 'affinity', second_data_source=True,
                                 filter_actives_test=test2_y_true, filter_actives_train=train2_y_true)
