from datetime import datetime


class Logger(object):
    def __init__(self, path_log_dir):
        filename = datetime.now().strftime("%m-%d-%Y,%H:%M:%S")
        self.file_writer = open(path_log_dir+"/"+filename+".txt", 'w')

    def write_record(self, transform):
        transform_flattened = transform[:3,:].flatten()
        record = str(transform_flattened.tolist())[1:-1].replace(",","") + "\n"
        self.file_writer.write(record)

    def close(self):
        self.file_writer.close()
