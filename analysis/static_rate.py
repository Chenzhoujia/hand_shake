import numpy as np

num_exampe = 2302

is_RNN = True

if is_RNN:

    total_tremor = np.zeros((num_exampe, 3))
    left_tremor = np.zeros((num_exampe, 3))

    for examp_i in range(num_exampe):

        wave_target = np.loadtxt("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test_result/RNN/"
                   + "target/" + str(examp_i).zfill(5) + ".txt")
        wave_result = np.loadtxt("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test_result/RNN/"
                   + "result/" + str(examp_i).zfill(5) + ".txt")

        wave_target = wave_target[100:, :]
        wave_result = wave_result[100:, :]
        for tmp_over1_i in range(len(wave_target)):
            for xyz_i in range(3):
                #累加振幅
                total_tremor[examp_i,xyz_i]+=abs(wave_target[tmp_over1_i,xyz_i])
                #计算有效的消除震颤的部分——附加后震颤幅度变小：方向一致且幅度小于2×target
                if wave_target[tmp_over1_i,xyz_i]*wave_result[tmp_over1_i,xyz_i]>0:
                    if abs(wave_result[tmp_over1_i, xyz_i]) < abs(wave_target[tmp_over1_i, xyz_i]):
                        left_tremor[examp_i,xyz_i]+=abs(wave_target[tmp_over1_i, xyz_i]) - abs(wave_result[tmp_over1_i, xyz_i])
                    elif abs(wave_result[tmp_over1_i, xyz_i])/2 < abs(wave_target[tmp_over1_i, xyz_i]):
                        left_tremor[examp_i,xyz_i]+=2*abs(wave_target[tmp_over1_i, xyz_i]) - abs(wave_result[tmp_over1_i, xyz_i])

        print(total_tremor[examp_i, :])
        print(left_tremor[examp_i, :])
    mean_rate = np.mean(left_tremor/total_tremor)
    print(mean_rate)#0.377378796674
else:

    total_tremor = np.zeros((num_exampe, 3))
    left_tremor = np.zeros((num_exampe, 3))

    for examp_i in range(num_exampe):

        wave_target = np.loadtxt("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test_result/wave/"
                   + "target/" + str(examp_i).zfill(5) + ".txt")
        wave_result = np.loadtxt("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test_result/wave/"
                   + "result/" + str(examp_i).zfill(5) + ".txt")

        wave_target = wave_target[100:, :]
        wave_result = wave_result[100:, :]
        for tmp_over1_i in range(len(wave_target)):
            for xyz_i in range(3):
                #累加振幅
                total_tremor[examp_i,xyz_i]+=abs(wave_target[tmp_over1_i,xyz_i])
                #计算有效的消除震颤的部分——附加后震颤幅度变小：方向一致且幅度小于2×target
                if wave_target[tmp_over1_i,xyz_i]*wave_result[tmp_over1_i,xyz_i]>0:
                    if abs(wave_result[tmp_over1_i, xyz_i]) < abs(wave_target[tmp_over1_i, xyz_i]):
                        left_tremor[examp_i,xyz_i]+=abs(wave_target[tmp_over1_i, xyz_i]) - abs(wave_result[tmp_over1_i, xyz_i])
                    elif abs(wave_result[tmp_over1_i, xyz_i])/2 < abs(wave_target[tmp_over1_i, xyz_i]):
                        left_tremor[examp_i,xyz_i]+=2*abs(wave_target[tmp_over1_i, xyz_i]) - abs(wave_result[tmp_over1_i, xyz_i])

        print(total_tremor[examp_i, :])
        print(left_tremor[examp_i, :])
    mean_rate = np.mean(left_tremor/total_tremor)
    print(mean_rate)#0.408934061455

    #np.savetxt("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test_result/RNN/"
    #            +"wave_rate.txt",wave_rate)

