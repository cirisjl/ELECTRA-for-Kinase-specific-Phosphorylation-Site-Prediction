import os
import sys
import pandas as pd
import numpy as np
import argparse
import csv
from methods.EXtractfragment_sort import extractFragforPredict

def main():
    parser=argparse.ArgumentParser(description='ELECTRA prediction tool for general, kinase-specific phosphorylation prediction.')
    parser.add_argument('-input', dest='inputfile', type=str, help='Protein sequences to be predicted in fasta format.', required=True)
    parser.add_argument('-predict-type',  
                        dest='predicttype', 
                        type=str, 
                        help='predict types. \'general\' for general human phosphorylation prediction by pre-trained ELECTRA model. \n \
                        \'kinase\' for kinase-specific human phosphorylation prediction by pre-trained ELECTRA models.\n \
                        It indicates two files [-model-prefix]_HDF5model and [-model-prefix]_parameters.', required=False)
    parser.add_argument('-kinase', dest='kinase', type=str, help='if -predict-type is \'kinase\', -kinase indicates the specific kinase, currently we accept \'CDK\' or \'PKA\' or \'CK2\' or \'MAPK\' or \'PKC\'.', required=False,default=None)
    parser.add_argument('-residue-types', dest='residues', type=str, help='Residue types that to be predicted, only used when -predict-type is \'general\'. For multiple residues, seperate each with \',\'',required=False,default="S,T")
    
    args = parser.parse_args()
    
    kinaselist=["CDK","PKA","CK2","MAPK","PKC"]
    
    inputfile=args.inputfile
    predicttype=args.predicttype
    residues=args.residues.split(",")
    kinase=args.kinase
    testpath='data/finetuning_data/'
    window=16
    
    if predicttype == 'general': #prediction for general phosphorylation
        if("S" in residues or "T" in residues):
            print("General phosphorylation prediction for S or T:\n")        
            testfrag,ids,poses,focuses=extractFragforPredict(inputfile,window,'-',focus=residues) 
            testlabel = testfrag[0]
            testfrag = testfrag[range(1,34)].apply(' '.join, axis = 1)
            testset=np.column_stack((testlabel, testfrag, ids, poses, focuses))
            testset=pd.DataFrame(testset)
            testset.to_csv(testpath + "ST/dev.tsv", index=False, header=None, sep='\t')
            
            os.system('python3 run_finetuning.py \
            --data-dir data \
            --model-name phosphorylation \
            --hparams \'{"model_size": "small", "do_train": false,"do_eval": true,"task_names": ["ST"]}\'' 
            )
              
            os.system('rm -rf ' + testpath + 'ST/*')
            os.system('rm -rf data/models/protein_small_quater_1m/finetuning_tfrecords/*')
            
        print("Successfully predicted for general phosphorylations sites!\n")
            
    elif predicttype == 'kinase':
        if kinase is None or kinase not in kinaselist:
            print("wrong parameter for -kinase! Must be one of \'CDK\' or \'PKA\' or \'CK2\' or \'MAPK\' or \'PKC\' !\n")
            exit()
        else: #prediction for kinas
            print("Kinase-specific prediction for "+str(kinase)+":\n")
            testfrag,ids,poses,focuses=extractFragforPredict(inputfile,window,'-',focus=("S","T")) 
            testlabel = testfrag[0]
            testfrag = testfrag[range(1,34)].apply(' '.join, axis = 1)
            testset=np.column_stack((testlabel, testfrag, ids, poses, focuses))
            testset=pd.DataFrame(testset)
            testset.to_csv(testpath + kinase + "/dev.tsv", index=False, header=None, sep='\t')
            
            os.system('python3 run_finetuning.py \
            --data-dir data \
            --model-name phosphorylation \
            --hparams \'{"model_size": "small", "do_train": false,"do_eval": true,"task_names": ["' + kinase + '"]}\'' 
            )
            
            os.system('rm -rf ' + testpath + kinase + '/*')
            os.system('rm -rf data/models/phosphorylation/finetuning_tfrecords/*')

        print("Successfully predicted kinase-specific phosphorylation sites for "+str(kinase)+"!\n")
        
    else: 
        print("wrong parameter for -predict-type!\n")
        exit()
        
        
if __name__ == "__main__":
    main()  
        
