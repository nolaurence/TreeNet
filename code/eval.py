import torch
from preprocess import data_slice


def evaluation(net, sample_length, x, y):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('working on evaluation ...')

    with torch.no_grad():
        for i in range(len(x)):
            inputs = torch.from_numpy(data_slice(x[i], sample_length)).to(device)
            labels = y[i]
            resident_precision = 0
            activity_precision = 0

            n_right_resident = 0
            n_right_activity = 0
            for j in range(inputs.shape[0]):
                outputs = net(inputs[j].view(8, 37))
                outputs = outputs.cpu().numpy()

                resident = list(outputs[:2])
                activity = list(outputs[2:])
                resident_gt = list(labels[j][:2])
                activity_gt = list(labels[j][2:])
                sorted_resident, sorted_activity = resident, activity
                sorted_resident.sort()
                sorted_activity.sort()

                if resident.index(sorted_resident[-1]) == resident_gt.index(1.0):
                    n_right_resident += 1
                if activity.index(sorted_activity[-1]) == activity_gt.index(1.0):
                    n_right_activity += 1
            resident_precision += n_right_resident / inputs.shape[0]
            activity_precision += n_right_activity / inputs.shape[0]
            # print("\nfile {}: resident's precision: {:.2f} %, activity's precision: {:.2f} %".format(
            #     i + 1, n_right_resident / inputs.shape[0]*100, n_right_activity / inputs.shape[0]*100))
        print("\nresident's precision: {:.2f} %, activity's precision: {:.2f} %".format(
            resident_precision / len(x) * 100, activity_precision / len(x) * 100))
