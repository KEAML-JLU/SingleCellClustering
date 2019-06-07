import csv

def process_line(line):
    line = line.split()
    file_name = line[0]
    acc = float(line[2])
    nmi = float(line[4])
    ari = float(line[6])
    tmp = file_name.split('/')
    dr_method = tmp[1]
    cl_method = tmp[2]
    if '_' in dr_method:
        dr_dim = int(dr_method.split('_')[1])
        dr_method = dr_method.split('_')[0]
    else:
        dr_dim = ''

    if '_' in cl_method:
        cl_dim = int(cl_method.split('_')[1])
        cl_method = cl_method.split('_')[0]
    else:
        cl_dim = ''

    return {
            'dr_method': dr_method,
            'dr_dim': dr_dim,
            'cl_method': cl_method,
            'cl_dim': cl_dim,
            'acc': acc,
            'nmi': nmi,
            'ari': ari}

with open('results.txt') as f, open('scores.csv','w') as fw:
    writer = csv.writer(fw)
    writer.writerow([
        'dr_method',
        'dr_dim',
        'cl_method',
        'nn_num',
        'acc',
        'nmi',
        'ari'])
    for l in f:
        l = l.strip()
        if l:
            tmp = process_line(l)
            row = [
                    tmp['dr_method'],
                    tmp['dr_dim'],
                    tmp['cl_method'],
                    tmp['cl_dim'],
                    tmp['acc'],
                    tmp['nmi'],
                    tmp['ari']]
            writer.writerow(row)
