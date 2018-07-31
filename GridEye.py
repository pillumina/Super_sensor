import time
import serial
import sys
import os

if __name__ == "__main__":
    for num in range(1, len(sys.argv)):
        print ("parameter: %s " % (sys.argv[num]))
    if len(sys.argv) != 3:
        print('Invalid number of parameters')
        exit(0)
    else:
        if sys.argv[1][1] != 'Y'and sys.argv[1][1] != 'N':
            print('Invalid parameters')
            exit(0)
        if sys.argv[2][1:].isdigit() == False:
            print('Invalid parameters')
            exit(0)

    status = sys.argv[1][1]
    datatime = int(sys.argv[2][1:])
    print ("status: %c " % (status ))
    print ("time: %d " % (datatime))
    startTime = int(time.time())
    lastTime = 0

    #
    ser = serial.Serial("/dev/ttyUSB0",
                        baudrate=115200,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS,
                        timeout=1)
						
    count = 0
    if os.path.exists("N") == False:
        os.mkdir("N")
    if os.path.exists("Y") == False:
        os.mkdir("Y")

    fname = os.getcwd() + "/" + status + "/" + "GE_" + status + str(int(round(time.time() * 1000))) + '.txt'
    print('start')
    f = open(fname, "ab")
    while 1:
        data = ser.read_all()
        length = len(data)
        if length != 0:
            f = open(fname, "ab")
            for i in range(0, length):
                flag_succes = 0

                if data[i:i+4] == b'$$$$':
                    index1 = 0
                    index2 = 0

                    for j in range(i, length):
                        if data[j:j + 4] == b'####':
                            index1 = j
                        if data[j:j + 4] == b'@@@@':
                            index2 = j
                        if index1 != 0 and index2 != 0:
                            if (index2 - i) < 500:
                                flag_succes = 1
                                break
                    if flag_succes == 1:
                        for k in range(i+4,index1):
                            if data[k] < 0x30 and data[k] > 0x39:
                                if data[k] != '.' and data[k] != ' ' and  data[k] != '\n':
                                    flag_succes = 0
                    if flag_succes == 1:
                        count = count + 1
                        if count > 600:
                            count = 1
                            f.close()
                            if datatime == 0:
                                print('end')
                                ser.close()
                                exit(0)
                            fname = os.getcwd() + "/"  + status + "/" + "GE_" + status + str(int(round(time.time() * 1000))) + '.txt'
                            f = open(fname, "ab")
                       # print('read:',data[i:index1])
                        f.write('Pixel Output:'+ '\n')
                        f.write(data[i+4:index1])
                        i = i + 480
                        f.write('Thermistor Temp: ')
                        f.write(data[index1 + 4:index2])
                        f.write(' \r\n\n')
            f.close()
        time.sleep(1)
        currentTime = int(time.time())
        if (currentTime - startTime - lastTime*60) > 60:
            lastTime = lastTime + 1
            datatime = datatime - 1
            print("time: %d " % (datatime))

