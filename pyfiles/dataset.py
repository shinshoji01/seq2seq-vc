import os
import numpy as np
import glob
from torch.utils.data import Dataset
from itertools import groupby

def transform(mel, defileprob=0.08, defilespan=10, defiling_ratio=[0.45, 0.45, 0.10]):
    # defiling ratio: [masking, noising, nothing]
    length = mel.shape[0]-defilespan
    if length>0:
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
    def __init__(self, feat_base_dir, dataset_dir, scaler, mode="train", limitlength=450, defiling_ratio=[0.45,0.45,0.10], input_output_type=["mel", "mel"], hubertnorepeating=False):
        modename = "dev" if mode=="valid" else mode
        inputname = "" if input_output_type[0]=="mel" else f"_{input_output_type[0]}"
        if inputname=="_hubert":
            inputname = "_hubert_km500"
        outputname = "" if input_output_type[1]=="mel" else f"_{input_output_type[1]}"

        speakers = [os.path.basename(a) for a in glob.glob(dataset_dir + "*/*")]
        speakers.sort()
        files = []
        for spk in speakers:
            for pt in glob.glob(feat_base_dir + f"{modename}*/{spk}/*/*[0-9]{inputname}.npy"):
                if inputname=="":
                    if "_hubert_km500" in pt:
                        continue
                files += [pt]
        ifiles = []
        ofiles = []
        for path in files:
            opath = os.path.dirname(path) + "/" + os.path.basename(path)[:-(4+len(inputname))]+f"{outputname}.npy"
            if os.path.exists(opath):
                ifiles += [path]
                ofiles += [opath]
            
        self.ifiles = ifiles
        self.files = ofiles
        self.scaler = scaler
        self.mode = mode
        self.limitlength = limitlength
        self.defiling_ratio = defiling_ratio
        self.input_output_type = input_output_type
        self.hubertnorepeating = hubertnorepeating
        
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        
        inputpath = self.ifiles[idx]
        outputpath = self.files[idx]
        add = "" if self.input_output_type[1]=="mel" else f"_{self.input_output_type[1]}"
        accpath = outputpath[:-(len(add)+4)] + "_accentembedding.npy"
        input_type = self.input_output_type[0]
        if input_type=="hubert":
            inputfeat = np.load(inputpath)
        else:
            inputfeat = self.scaler[0].transform(np.load(inputpath).T)
        outputfeat = self.scaler[1].transform(np.load(outputpath).T)
        if outputfeat.shape[0]>self.limitlength:
            start = np.random.randint(0, outputfeat.shape[0]-self.limitlength)
            end = start+self.limitlength
            startinput = int(inputfeat.shape[0]*start/outputfeat.shape[0])
            endinput = int(inputfeat.shape[0]*end/outputfeat.shape[0])
            inputfeat = inputfeat[startinput:endinput]
            outputfeat = outputfeat[start:end]
        acc = np.load(accpath)
        if input_type=="hubert":
            inputfeat = list(inputfeat.astype(int))
            if self.hubertnorepeating:
                inputfeat = [key for key, _group in groupby(inputfeat)]
            inputfeat = np.array(inputfeat).reshape(-1, 1).astype(int)

        items = {}
        if input_type in ["mel", "80mel"]:
            items["src_feat"] = transform(inputfeat.copy(), defiling_ratio=self.defiling_ratio)
        else:
            items["src_feat"] = inputfeat
        items["trg_feat"] = outputfeat
        items["src_condition"] = acc
        items["trg_condition"] = acc
        items["accent_id"] = 0
        
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
        items["accent_id"] = 0
        
        return items
    
class PretrainingL2Arctic(Dataset):
    def __init__(self, dataset_dir, speakers, accents, genders, datasplit, scaler, spk2acc, spk2sex, mode="train", input_output_type=["wavlm", "mel"]):
        modefiles = datasplit[["train", "valid", "test"].index(mode)]
        data = {}
        for spk in speakers:
            data[spk] = []
            for a in glob.glob(dataset_dir + f"{spk}/{input_output_type[0]}/*.npy"):
                basename = os.path.basename(a)[:-4]
                if basename in modefiles:
                    data[spk] += [basename]
            data[spk].sort()
                
        files = []
        for s, spk in enumerate(speakers):
            accentid = accents.index(spk2acc[spk])
            genderid = genders.index(spk2sex[spk])
            for basename in data[spk]:
                files += [[s, accentid, genderid, basename]]
            
        self.dataset_dir = dataset_dir
        self.data = data
        self.scaler = scaler
        self.files = files
        self.speakers = speakers
        self.accents = accents
        self.genders = genders
        self.input_output_type = input_output_type
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        speaker, accent, gender, basename = self.files[idx]
        items = {}
        src_mel = self.dataset_dir + f"{self.speakers[speaker]}/{self.input_output_type[0]}/{basename}.npy"
        trg_mel = self.dataset_dir + f"{self.speakers[speaker]}/{self.input_output_type[1]}/{basename}.npy"
        items["src_feat"] = self.scaler[self.input_output_type[0]].transform(np.load(src_mel).T)
        items["trg_feat"] = self.scaler[self.input_output_type[1]].transform(np.load(trg_mel).T)
        items["src_condition"] = np.load(src_mel.replace(self.input_output_type[0], "accent_embedding"))
        items["trg_condition"] = np.load(trg_mel.replace(self.input_output_type[1], "accent_embedding"))
        items["utt_id"] = basename
        items["speaker_id"] = speaker
        items["accent_id"] = accent
        items["gender_id"] = gender
        
        return items
    
class ParallelArcticDataset(Dataset):
    def __init__(self, src_dir, trg_dir, datasplit, scaler, mode="train", input_output=["wavlm", "mel"], noembedding=False):
        if type(src_dir)!=list:
            src_dir = [src_dir]
            trg_dir = [trg_dir]
        assert len(src_dir)==len(trg_dir)
        modefiles = datasplit[["train", "valid", "test"].index(mode)]
        
        data = {}
        data["src"] = []
        data["trg"] = []
        allfiles = []
        for diridx in range(len(src_dir)):
            src = src_dir[diridx]
            trg = trg_dir[diridx]
            
            filenames = [os.path.basename(a)[:-4] for a in glob.glob(src+f"{input_output[0]}/*.npy")]
            filenames.sort()
            files = []
            for fn in filenames:
                if fn in modefiles:
                    if os.path.exists(f"{trg}{input_output[1]}/{fn}.npy"):
                        files += [fn]
            allfiles += files
            print(len(files), src)
            data["src"] += [src + f"{input_output[0]}/{fn}.npy" for fn in files]
            data["trg"] += [trg + f"{input_output[1]}/{fn}.npy" for fn in files]
            
        self.data = data
        self.scaler = scaler
        self.files = allfiles
        self.input_output = input_output
        self.noembedding = noembedding
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        items = {}
        src_mel = self.data["src"][idx]
        trg_mel = self.data["trg"][idx]
        items["src_feat"] = self.scaler[self.input_output[0]].transform(np.load(src_mel).T)
        items["trg_feat"] = self.scaler[self.input_output[1]].transform(np.load(trg_mel).T)
        try:
            items["src_condition"] = np.load(src_mel.replace(self.input_output[0], "accent_embedding"))
            items["trg_condition"] = np.load(trg_mel.replace(self.input_output[1], "accent_embedding"))
        except FileNotFoundError:
            assert self.noembedding
            items["src_condition"] = np.random.randn(1024)
            items["trg_condition"] = np.random.randn(1024)
        items["utt_id"] = self.files[idx]
        items["accent_id"] = 0
        
        return items