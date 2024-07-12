import os
import numpy as np
import glob
from torch.utils.data import Dataset

def transform(mel, defileprob=0.08, defilespan=10, defiling_ratio=[0.45, 0.45, 0.10]):
    # defiling ratio: [masking, noising, nothing]
    length = mel.shape[0]-defilespan
    bool_list = []
    for start in np.arange(length)[np.random.rand(length)<defileprob]:
        bool_list += list(range(start, start+defilespan))
    bool_list = np.array(list(set(bool_list)))

    if len(bool_list)>0:
        option = np.random.rand()
        if option<defiling_ratio[0]: # Masking
            mel[bool_list] = 0
        elif option>1-defiling_ratio[1]:
            mel[bool_list] = mel[bool_list] + np.random.randn(len(bool_list), mel.shape[1])*0.2
    return mel

class PretrainingMelDataset(Dataset):
    def __init__(self, feat_base_dir, dataset_dir, scaler, mode="train", limitlength=450, defiling_ratio=[0.45,0.45,0.10], input_output_type=["mel", "mel"]):
        modename = "dev" if mode=="valid" else mode
        speakers = [os.path.basename(a) for a in glob.glob(dataset_dir + "*/*")]
        speakers.sort()
        files = []
        for spk in speakers:
            files += glob.glob(feat_base_dir + f"{modename}*/{spk}/*/*[0-9].npy")
        files.sort()
            
        self.files = files
        self.scaler = scaler
        self.mode = mode
        self.limitlength = limitlength
        self.defiling_ratio = defiling_ratio
        self.input_output_type = input_output_type
        
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        
        outputpath = dataset.files[idx]
        accpath = outputpath[:-4] + "_accentembedding.npy"
        input_type = dataset.input_output_type[0]
        if input_type=="mel":
            inputpath = outputpath
        else:
            inputpath = outputpath[:-4] + f"_{input_type}.npy"
        inputfeat = dataset.scaler[0].transform(np.load(inputpath).T)
        outputfeat = dataset.scaler[1].transform(np.load(outputpath).T)
        if outputfeat.shape[0]>dataset.limitlength:
            start = np.random.randint(0, outputfeat.shape[0]-dataset.limitlength)
            end = start+dataset.limitlength
            startinput = int(inputfeat.shape[0]*start/outputfeat.shape[0])
            endinput = int(inputfeat.shape[0]*end/outputfeat.shape[0])
            inputfeat = inputfeat[startinput:endinput]
            outputfeat = outputfeat[start:end]
        acc = np.load(accpath)

        items = {}
        if input_type=="mel":
            items["src_feat"] = transform(inputfeat.copy(), defiling_ratio=dataset.defiling_ratio)
        else:
            items["src_feat"] = inputfeat
        items["trg_feat"] = outputfeat
        items["src_condition"] = acc
        items["trg_condition"] = acc
        
        return items

    
class Parallelo2oVCMelDataset(Dataset):
    def __init__(self, dataset_dir, speakers, src_spk, trg_spk, datasplit, scaler, mode="train", randomcondition=False):
        modefiles = datasplit[["train", "valid", "test"].index(mode)]
        filenames = [os.path.basename(a)[:-4] for a in glob.glob(dataset_dir+f"{speakers[0]}/mel/*")]
        filenames.sort()
        files = []
        for fn in filenames:
            if fn in modefiles:
                exist = True
                for spk in speakers[1:]:
                    if not(os.path.exists(dataset_dir + f"{spk}/mel/{fn}.npy")):
                        exist = False
                        break
                if exist:
                    files += [fn]
        data = {}
        for spk in speakers:
            data[spk] = [dataset_dir + f"{spk}/mel/{fn}.npy" for fn in files]
            
        self.data = data
        self.src_spk = src_spk
        self.trg_spk = trg_spk
        self.scaler = scaler
        self.randomcondition = randomcondition
        self.files = files
        
    def __len__(self):
        return len(self.data[self.src_spk])
    
    def __getitem__(self, idx):
        items = {}
        src_mel = self.data[self.src_spk][idx]
        trg_mel = self.data[self.trg_spk][idx]
        items["src_feat"] = self.scaler.transform(np.load(src_mel).T)
        items["trg_feat"] = self.scaler.transform(np.load(trg_mel).T)
        sm =  self.data[self.src_spk][np.random.randint(len(self.data[self.src_spk]))] if self.randomcondition else src_mel
        tm =  self.data[self.trg_spk][np.random.randint(len(self.data[self.trg_spk]))] if self.randomcondition else trg_mel
        items["src_condition"] = np.load(sm.replace("mel", "accent_embedding"))
        items["trg_condition"] = np.load(tm.replace("mel", "accent_embedding"))
        items["utt_id"] = self.files[idx]
        
        return items