class DataLogger:
    
    def __init__(self, args):
        self.file_log = open(args.save_dir + '/' + args.network_name[0] + '/lr_log.txt','w')
    
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
            
            