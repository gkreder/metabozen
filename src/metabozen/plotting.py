import os
import sys
import pandas as pd
from tqdm.notebook import tqdm
from pyteomics import mzml


class MassSpecProcessor:
    def __init__(self, excel_path, data_dir, mzml_dir, scraping_dir):
        self.excel_path = excel_path
        self.data_dir = data_dir
        self.mzml_dir = mzml_dir
        self.scraping_dir = scraping_dir
        self.xl = pd.ExcelFile(excel_path)
        self.snames = self.xl.sheet_names
        self.nd = {
            'Pos Mode': "pos",
            'Neg Mode': "neg",
            'Pos Mode Sulfatase': "pos_sulfatase",
            'Neg Mode Sulfatase': "neg_sulfatase",
            'Pos Addnl Stds': "pos_aStds",
            'Neg Addnl Stds': "neg_aStds"
        }
        self.ods = self.nd

    def get_fname(self, sn, i, std):
        dd = {
            'Pos Mode': f"{self.mzml_dir}/Pos_Mode/Urine/Untreated",
            'Neg Mode': f"{self.mzml_dir}/Neg_Mode/Urine/Untreated",
            'Pos Mode Sulfatase': f"{self.mzml_dir}/Pos_Mode/Urine/Sulfatase",
            'Neg Mode Sulfatase': f"{self.mzml_dir}/Neg_Mode/Urine/Sulfatase"
        }
        dds = {
            'Pos Addnl Stds': f"{self.mzml_dir}/Pos_Mode/Stds",
            'Neg Addnl Stds': f"{self.mzml_dir}/Neg_Mode/Stds",
            'Pos Mode': f"{self.mzml_dir}/Pos_Mode/Stds",
            'Neg Mode': f"{self.mzml_dir}/Neg_Mode/Stds",
            'Pos Mode Sulfatase': f"{self.mzml_dir}/Pos_Mode/Stds",
            'Neg Mode Sulfatase': f"{self.mzml_dir}/Neg_Mode/Stds"
        }
        pss = {
            'Pos Addnl Stds': "Pos_Std_MSMS",
            'Neg Addnl Stds': "Neg_Std_MSMS",
            'Pos Mode': "Pos_Std_MSMS",
            'Neg Mode': "Neg_Std_MSMS",
            'Pos Mode Sulfatase': "Pos_Std_MSMS",
            'Neg Mode Sulfatase': "Neg_Std_MSMS"
        } if std else {
            'Pos Mode': "Urine_C18_MSMS_Pos",
            'Neg Mode': "Urine_C18_MSMS_Neg",
            'Pos Mode Sulfatase': "Urine_C18_MSMS_Pos",
            'Neg Mode Sulfatase': "Urine_C18_MSMS_Neg"
        }
        d = dds[sn] if std else dd[sn]
        p = os.path.join(d, f"{pss[sn]}_{i}.mzML")
        if not os.path.exists(p):
            sys.exit(f"Given {sn} {i} {std} and fname is {p}")
        return p

    def get_out_fname(self, sn, n, v, std):
        dd = {
            'Pos Mode': f"{self.scraping_dir}/pos_untreated/mgfs",
            'Neg Mode': f"{self.scraping_dir}/neg_untreated/mgfs",
            'Pos Mode Sulfatase': f"{self.scraping_dir}/pos_sulfatase/mgfs",
            'Neg Mode Sulfatase': f"{self.scraping_dir}/neg_sulfatase/mgfs"
        }
        dds = {
            'Pos Addnl Stds': f"{self.scraping_dir}/pos_stds/mgfs",
            'Neg Addnl Stds': f"{self.scraping_dir}/neg_stds/mgfs",
            'Pos Mode': f"{self.scraping_dir}/pos_stds/mgfs",
            'Neg Mode': f"{self.scraping_dir}/neg_stds/mgfs",
            'Pos Mode Sulfatase': f"{self.scraping_dir}/pos_stds/mgfs",
            'Neg Mode Sulfatase': f"{self.scraping_dir}/neg_stds/mgfs"
        }
        d = dds[sn] if std else dd[sn]
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{n}_{v}V.mgf")

    def make_mgf(self, s, out_fname, rt):
        rt_found = str(s['scanList']['scan'][0]['scan start time'])
        array = list(zip(s['m/z array'], s['intensity array']))
        pepmass = str(s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])
        charge = f"{int(s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state'])}+"
        title = os.path.basename(out_fname).split('.')[0]
        with open(out_fname, 'w') as fout:
            fout.write("BEGIN IONS\n")
            fout.write(f"PEPMASS={pepmass}\n")
            fout.write(f"CHARGE={charge}\n")
            fout.write(f"TITLE={title}\n")
            fout.write(f"RT_FOUND={rt_found}\n")
            fout.write(f"RT_SOUGHT={rt}\n\n")
            for mz, inten in array:
                fout.write(f"{mz} {inten}\n")
            fout.write("\nEND IONS\n")

    def process_spectra(self):
        check_rows = []
        for sn in self.snames:
            pref = self.nd[sn]
            pref_dir = os.path.join(self.data_dir, pref)
            if 'Addnl' not in sn:
                for istd, std in enumerate([False, True]):
                    print(sn, std)
                    scn = ['Sample MS/MS Run', "Std. MS/MS Run"][istd]
                    cn = "Mode-specific feature name"
                    df = self.xl.parse(sn).dropna(subset=[scn])
                    df[scn] = df[scn].astype(str)
                    df = df.sort_values(by=scn)
                    vd = [{0: '0 V RT', 10: '10 V RT', 20: '20 V RT', 40: '40 V RT'},
                         {0: '0 V RT.1', 10: '10 V RT.1', 20: '20 V RT.1', 40: '40 V RT.1'}][istd]
                    prev_infile = ''
                    for i, pn in enumerate(tqdm(df[cn])):
                        imsms = df[scn].values[i]
                        if str(imsms) == 'nan' or str(imsms) == "Check":
                            continue
                        in_fname = self.get_fname(sn, imsms, std=std)
                        if in_fname != prev_infile:
                            f = mzml.MzML(in_fname)
                        prev_infile = in_fname
                        for v in [0, 10, 20, 40]:
                            out_fname = self.get_out_fname(sn, pn, v, std=std)
                            rt = df[vd[v]].values[i]
                            if str(rt) == 'nan':
                                continue
                            s = f.time[rt]
                            try:
                                _ = str(s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])
                            except:
                                print(f'MANUAL OVERRIDE FOR {pn} {v}V')
                                reader = mzml.read(in_fname)
                                spectra = [x for x in reader if x['ms level'] == 2]
                                sorted_spectra = sorted([(abs(float(x['scanList']['scan'][0]['scan start time']) - rt), x) for x in spectra], key=lambda tup: tup[0])
                                if abs(float(sorted_spectra[0][1]['scanList']['scan'][0]['scan start time']) - rt) > 0.003:
                                    sys.exit('RT check error')
                                if abs(float(sorted_spectra[0][1]['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']) - float(pn.split('T')[0][1:])) > 1.0:
                                    print('manual check error')
                                    continue
                                s = sorted_spectra[0][1]
                                print(f'BEST FOUND RT {str(s["scanList"]["scan"][0]["scan start time"])} MZ {str(s["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["selected ion m/z"])}')
                            self.make_mgf(s, out_fname, rt)
                            check_rows.append((sn, std, pn, v, rt, out_fname))
            else:
                std = True
                print(sn, std)
                scn = "Std. MS/MS Run"
                cn = "Assignment"
                df = self.xl.parse(sn).dropna(subset=[scn])
                df[scn] = df[scn].astype(str)
                df = df.sort_values(by=scn)
                vd = {0: '0 V RT', 10: '10 V RT', 20: '20 V RT', 40: '40 V RT'}
                prev_infile = ''
                for i, pn in enumerate(tqdm(df[cn])):
                    imsms = df[scn].values[i]
                    if str(imsms) == 'nan':
                        continue
                    in_fname = self.get_fname(sn, imsms, std=std)
                    if in_fname != prev_infile:
                        f = mzml.MzML(in_fname)
                    prev_infile = in_fname
                    for v in [0, 10, 20, 40]:
                        out_fname = self.get_out_fname(sn, pn, v, std=std)
                        rt = df[vd[v]].values[i]
                        if str(rt) == 'nan':
                            continue
                        s = f.time[rt]
                        try:
                            _ = str(s['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])
                        except:
                            print(f'MANUAL OVERRIDE FOR {pn} {v}V')
                            reader = mzml.read(in_fname)
                            spectra = [x for x in reader if x['ms level'] == 2]
                            sorted_spectra = sorted([(abs(float(x['scanList']['scan'][0]['scan start time']) - rt), x) for x in spectra], key=lambda tup: tup[0])
                            if abs(float(sorted_spectra[0][1]['scanList']['scan'][0]['scan start time']) - rt) > 0.003:
                                print('RT check error', sorted_spectra[0][1]['scanList']['scan'][0]['scan start time'], rt)
                                continue
                            if abs(float(sorted_spectra[0][1]['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']) - df[[x for x in df.columns if "m/z" in x][0]].values[i]) > 1.0:
                                print('manual check error', sorted_spectra[0][1]['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])
                                continue
                            s = sorted_spectra[0][1]
                            print(f'BEST FOUND RT {str(s["scanList"]["scan"][0]["scan start time"])} MZ {str(s["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["selected ion m/z"])}')
                        self.make_mgf(s, out_fname, rt)
                        check_rows.append((sn, std, pn, v, rt, out_fname))
        
        return check_rows


def main():
    excel_path = "/mnt/data/200529_MSMS_Spectra_RT_Information.xlsx"
    data_dir = "/media/gkreder/5TB/data/mass_spec/meyer/200528_MSMS_scraping"
    mzml_dir = "/media/gkreder/5TB/data/mass_spec/meyer/200521_Meyer_MSMS_mzML_Files"
    scraping_dir = "/media/gkreder/5TB/data/mass_spec/meyer/200528_MSMS_scraping"

    processor = MassSpecProcessor(excel_path, data_dir, mzml_dir, scraping_dir)
    check_rows = processor.process_spectra()

    # Additional processing can go here
    # For example, saving check_rows to a CSV file
    check_df = pd.DataFrame(check_rows, columns=["sn", "std", "pn", "v", "rt", "out_fname"])
    check_df.to_csv("check_rows.csv", index=False)

if __name__ == "__main__":
    main()
