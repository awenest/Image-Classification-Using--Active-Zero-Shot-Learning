import os

root_dir = "/home/rishabh/project/dataset/Animals_with_Attributes2/antelope/"
for root, subdirs, files in os.walk(root_dir):
    list_file_path = os.path.join(root, 'list_of_files.txt')
    with open(list_file_path, 'wb') as list_file:

        for filename in files:
            if filename.endswith("jpg"):
                file_path = os.path.join(root, filename)
                np_name = filename[0:-4]
                np_name = np_name+".npy"

                npy = open(os.path.join(root,np_name),"w+")
        #    print('file %s (full path: %s)' % (filename, file_path))
                with open(file_path, 'rb') as f:
                    list_file.write(('%s\n' % filename).encode('utf-8'))



