import os
import numpy


def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))

# print(sorted_ls('documents'))

# current_files = os.listdir(os.getcwd())
current_files = sorted_ls(os.getcwd())
for f_name in current_files:
    if f_name.endswith('.npz'):
        with open(f_name,'r') as f:
            print(f_name)
            content = numpy.load(f)
            hist = content['history_errs']

            # print hist
            l = (len(hist))
            with open(f_name+'.dat','w') as wtf:
                wtf.write('Iteration\tAccuracy\tMSE\n')
                for i in xrange(len(hist)):
                    wtf.write('%s\t%s\t%s\n'%(i,round((1-hist[i][0])*100,1),round(hist[i][1],5)))
            print('----Done----')