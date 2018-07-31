import time
import serial
import sys
import os

if  __name__ == "__main__":
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
    print("status: %c " % (status))
    print("time: %d " % (datatime))
    startTime = int(time.time())
    lastTime = 0

    # / dev / ttyUSB5
    ser = serial.Serial("/dev/ttyUSB0",
                        baudrate=115200,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS,
                        timeout=1)

    count = 0
    if os.path.exists("PIR_N") == False:
        os.mkdir("PIR_N")
    if os.path.exists("PIR_Y") == False:
        os.mkdir("PIR_Y")

    fname = os.getcwd() + "/" + "PIR_" + status + "/" + "PIR_" + status + str(int(round(time.time() * 1000))) + '.txt'
    print('start')
    f = open(fname, "ab")

    while 1:
        out = 0
        data = ser.read_all()
        length = len(data)
        if length != 0:
            for i in range(0, length):
                if data[i:i + 4] == b'Core':
                    log = data[i-4:i-2]
                    # log = int(log.strip())
                    print(log)
                    out = 1
                    break
            if out == 1:
                break




    while 1:
        # print("Main loop")
        data = ser.read_all()
        length = len(data)
        if length != 0:
            f = open(fname, "ab")
            for i in range(0, length):
                flag_succes = 0

                if data[i:i+4] == b'curr':
                    flag_succes = 1
                if flag_succes == 1:
                    count = count + 1
                    if count > 600:
                        count = 1
                        f.close()
                        if datatime == 0:
                            print('end')
                            ser.close()
                            exit(0)
                        fname = os.getcwd() + "/" + "PIR_" + status + "/" + "PIR_" + status + str(int(round(time.time() * 1000))) + '.txt'
                        f = open(fname, "ab")
                    f.write(data[i+6:i+10])
                    i = i + 9
                    f.write(b'\n')
            f.close()
        time.sleep(1)
        currentTime = int(time.time())
        if (currentTime - startTime - lastTime * 60) > 60:
            lastTime = lastTime + 1
            datatime = datatime - 1
            print("time: %d " % (datatime))

