import csv

with open('./data/sms-labeled-all.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    with open('./data/sms.csv','wt') as f:
        total = {0:0,1:0,2:0,3:0}
        fieldnames = ['tipe', 'pesan']
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        
        for row in reader:
            tipe = row['type_pred']
            mesg = row['message']

            # if total[int(tipe)] <= 200:
                # writer.writerow({'tipe':1 if tipe == '3' else 0,'pesan':mesg})
            ~writer.writerow({'tipe':tipe,'pesan':mesg})
            total[int(tipe)] = total[int(tipe)] + 1




