##Stroker.py
##Elvin Lin
##Anthony Thai

import csv

class Patient:
    def __init__(identifier,id, gender, age, hypertension, heart_disease, ever_married, work_type, residence, avg_glucose_levels, bmi, smoking_status, stroke):
        identifier.id = id
        identifier.gender = gender
        identifier.age = age
        identifier.hypertension = hypertension
        identifier.heart_disease = heart_disease
        identifier.ever_married = ever_married
        identifier.work_type = work_type
        identifier.residence = residence
        identifier.avg_glucose_levels = avg_glucose_levels
        identifier.bmi = bmi
        identifier.smoking_status = smoking_status
        identifier.stroke = stroke
    
    
def doWork():
    pL = []
    patientList = []
    with open('healthcare-dataset-stroke-data.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            pL.append(lines)
        del pL[0]
        
    for i in range (len(pL)):
        patientList.append(pL[i][0])
        pid     = pL[i][0]
        pgender = pL[i][1]
        page    = float(pL[i][2])
        phypert = pL[i][3]
        pheart_ = pL[i][4]
        pever_m = pL[i][5]
        pwork_t = pL[i][6]
        preside = pL[i][7]
        pavg_gl = pL[i][8]
        if pL[i][9] != "N/A":
            pbmi = pL[i][9]
        else:
            print('checked')
            if page >= 2 and page <= 5:
                pbmi == 16.5
            if page >= 6 and page <= 7:
                pbmi == 17.2
            if page >= 8 and page <= 9:
                pbmi == 18.3
            if page >= 10 and page <= 11:
                pbmi == 20.2
            if page >= 12 and page <= 13:
                pbmi == 22.0
            if page >= 14 and page <= 15:
                pbmi == 23.4
            if page >= 16 and page <= 17:
                pbmi == 25.3
            if page >= 18 and page <= 24:
                pbmi == 27.1
            if page >= 25 and page <= 29:
                pbmi == 27.9
            if page >= 30 and page <= 34:
                pbmi == 29.6
            if page >= 35 and page <= 39:
                pbmi == 30.2
            if page >= 30 and page <= 44:
                pbmi == 30.1
            if page >= 45 and page <= 49:
                pbmi == 29.7
            if page >= 50 and page <= 54:
                pbmi == 30.1
            if page >= 55 and page <= 59:
                pbmi == 29.8
            if page >= 60 and page <= 64:
                pbmi == 30.5
            if page >= 65 and page <= 69:
                pbmi == 30.0
            if page >= 70 and page <= 74:
                pbmi == 29.8
            if page >= 75:
                pbmi == 28.1
        psmokin = pL[i][10]
        pstroke = pL[i][11]
        #streamlining age
        page = int(round(page))
        patientList[i] = Patient(pid,pgender, page, phypert, pheart_, pever_m, pwork_t, preside, pavg_gl, pbmi, psmokin,pstroke)

        return(patientList)
if __name__ == "__main__":
    doWork()
    