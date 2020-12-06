from datetime import datetime
from factor_graph_utils import X
import os

class Logger(object):
    local_dir_name = None
    dir_path = None
    def __init__(self, config, prefix=""):
        '''
        Initialise the writer
        '''
        if Logger.local_dir_name is None:
            Logger.create_directory(config)
        # save file and dir paths
        self.file_name = Logger.dir_path+"/"+prefix+"log.txt" 
        self.file_writer = open(self.file_name, 'w')

        # frame ids and gt_file paths
        self.start = config["start_frame_num"]+1
        self.end = config["length_traj"] + self.start - 1
        self.gt_file = config["gt_dir"] + "/" + config["seq"] + ".txt"

    def write_record(self, transform):
        transform_flattened = transform[:3,:].flatten()
        record = str(transform_flattened.tolist())[1:-1].replace(",","") + "\n"
        self.file_writer.write(record)

    def close(self):
        self.file_writer.close()
        self.eval()

    def write_factor_graph(self, values):
        length = values.size()
        for i in range(length):
            val = values.atPose3(X(i))
            transform_flattened = val.matrix()[:3,:].flatten()
            record = str(transform_flattened.tolist())[1:-1].replace(",","") + "\n"
            self.file_writer.write(record)

    def eval(self):
        '''
        Call evo ape here with just the file having top few records
        '''
        data = None
        local_gt_path = Logger.dir_path + "/gt.txt"
        with open(self.gt_file, 'r') as fin:
            data = fin.read().splitlines(True)

        with open(local_gt_path, 'w') as fout:
            fout.writelines(data[self.start:self.end])

        result_file_path = Logger.dir_path + "/result.txt"
        command = "touch {0}".format(result_file_path)
        os.system(command)

        command= "evo_ape kitti -r trans_part --plot --plot_mode xz --logfile {0} {1} {2}"\
            .format(result_file_path, local_gt_path, self.file_name) 
        os.system(command)

    @classmethod
    def create_directory(cls, config):
        path_log_dir = config["log_dir"]
        local_dir_name = datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
        os.makedirs(path_log_dir+"/"+local_dir_name)
        Logger.local_dir_name = local_dir_name
        Logger.dir_path = path_log_dir+"/"+local_dir_name
