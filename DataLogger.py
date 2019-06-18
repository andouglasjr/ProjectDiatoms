import numpy as np
import os
np.set_printoptions(threshold=np.nan)


class DataLogger:
    
    __instance = None
    
    @staticmethod
    def getInstance(args):
        if DataLogger.__instance == None:
            DataLogger(args)
        return DataLogger.__instance
    
    def __init__(self, args):
        DataLogger.__instance = self
        folder = args.save_dir+'/'+str(args.network_name[0])+'/logs'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.file_log = open(folder+'/'+str(args.time_training)+'.txt','w')
    
    def log(self, content, log, close = False):
        if(log == 'l'):
            print("[!] {}".format(content))
            self.file_log.write("[!] {}\n".format(content))
        elif(log == 'v'):
            print("[v] {}".format(content))
            self.file_log.write("[!] {}\n".format(content))
        elif(log == 'e'):
            print("----> {}".format(content))
            self.file_log.write("----> {}\n".format(content))
        if close:
            self.file_log.close()
            
            