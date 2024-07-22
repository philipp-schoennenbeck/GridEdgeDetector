import click
from grid_edge_detector.image_gui import main, load_config_file
from grid_edge_detector.carbon_edge_detector import find_grid_hole_per_file, load_file,suffix_function_dict
import toml
import os
from pathlib import Path
import glob
import multiprocessing as mp
import numpy as np
import mrcfile
from matplotlib import pyplot as plt
from cryosparc.tools import Dataset
import shutil
import tqdm
import starfile

@click.group()
def run_ged():
    pass



@run_ged.command()
def GUI():
    main()



def exportCryosparc(jobPath, ds, particles, path_translator):


    ds = ds = Dataset.load(jobPath / "picked_particles.cs")
    idxs = []

    for counter, (x,y, m) in enumerate(zip(ds["location/center_x_frac"], ds["location/center_y_frac"], ds["location/micrograph_path"])):
        if m in path_translator and path_translator[m] in particles and particles[path_translator[m]]["mask"] is not None:
            shape = particles[path_translator[m]]["shape"] 
            
            x = int(x*shape[1])
            y = int(y*shape[0])
            if  particles[path_translator[m]]["mask"][y,x] == 1:
                idxs.append(True)
            else:
                idxs.append(False)
        else:
            idxs.append(True)
    ds = ds.mask(idxs)
    if not (jobPath / "picked_particles_old.cs").exists():
        shutil.move(jobPath / "picked_particles.cs", jobPath / "picked_particles_old.cs")
    else:
        mv_path = jobPath / "picked_particles_old.cs"
        print(f"{mv_path} does already exist. Probably from a previous masking job.\nWill not overwrite it to preserve the original coordinates.\nIf you want to run this on the original coordinates, remove the \"_old\" from this file. ")
    ds.save(jobPath / "picked_particles.cs")


def is_valid_job_path(path):
    return (path / "picked_micrographs.cs").exists() and (path / "picked_particles.cs").exists()


def run_cryosparc(config):
    if not is_valid_job_path(Path(config["command_line_arguments"]["cryosparc"]["job_path"])):
        print(config["command_line_arguments"]["cryosparc"]["job_path"] + " is not a valid cryosparc job (picked_particles.cs and/or picked_micrographs.cs were not found.)")
        exit()
    if config["command_line_arguments"]["masked"]["save"] or config["command_line_arguments"]["mask"]["save"]:
        response = input("While running a cryosparc job masking \"masked.save\" or/and \"mask.save\" is true. Do you want to save files additionally to masking picked particles from cryosparc? [y/n]")
        if response != "y":
            config["command_line_arguments"]["masked"]["save"] = False
            config["command_line_arguments"]["mask"]["save"] = False

    jobPath = Path(config["command_line_arguments"]["cryosparc"]["job_path"])
    project_folder = jobPath.parent
    ds = Dataset.load(jobPath / "picked_particles.cs")
    paths = sorted(list(set(ds["location/micrograph_path"])))
    particles = {}
    path_translator = {}
    for key in paths:
        current_file = project_folder / key
        if current_file.is_symlink():
            current_file = current_file.readlink()
        particles[current_file] = {"mask":None, "shape":None, "ps":None}
        path_translator[key] = current_file




    ds = Dataset.load(jobPath / "picked_micrographs.cs")
    for ps, m, shape in zip(ds["micrograph_blob/psize_A"], ds["micrograph_blob/path"], ds["micrograph_blob/shape"]):
        
        particles[path_translator[m]]["ps"] = ps
        particles[path_translator[m]]["shape"] = shape
    
    files = [key for key in sorted(particles.keys())]

    
    if config["parameters"]["njobs"] <= 1:
        for file in tqdm.tqdm(files):
            particles[file]["mask"] = run_masking(file, config)
    else:
        with mp.get_context("spawn").Pool(config["parameters"]["njobs"]) as pool:
            results = [pool.apply_async(run_masking, args=[file, config]) for file in files]
            result = []
            for res in tqdm.tqdm(results):
                result.append(res.get())
            # result = [res.get() for res in result]
            for mask, file in zip(result, files):
                particles[file]["mask"] = mask
    
    exportCryosparc(jobPath, ds, particles, path_translator)

def is_valid_relion_picking_job(path):
    return path.exists() and path.name in ["autopick.star", "manualpick.star"]

def run_relion(config):
    if not is_valid_relion_picking_job(Path(config["command_line_arguments"]["relion"]["picking_path"])):
        print(config["command_line_arguments"]["relion"]["picking_path"] + " is not a valid relion picking star file (it does not exist or is not an autopick.star/manualpick.star file.)")
        exit()

    if config["command_line_arguments"]["masked"]["save"] or config["command_line_arguments"]["mask"]["save"]:
        response = input("While running a relion job masking \"masked.save\" or/and \"mask.save\" is true. Do you want to save files additionally to masking picked particles from relion? [y/n]")
        if response != "y":
            config["command_line_arguments"]["masked"]["save"] = False
            config["command_line_arguments"]["mask"]["save"] = False

    def extract_coordinates(sf):
            coordinates_df = starfile.read(sf)
            x = coordinates_df["rlnCoordinateX"].astype(np.int32)
            y = coordinates_df["rlnCoordinateY"].astype(np.int32)
            return y,x

    pick_file = Path(config["command_line_arguments"]["relion"]["picking_path"])
    project_file = pick_file.parent.parent.parent
    autopick_df = starfile.read(pick_file)

    rlnInfo = {}
    files = []


    for i in range(len(autopick_df)):

        absolut_path = project_file/ autopick_df.loc[i]["rlnMicrographName"]
        rlnInfo[absolut_path] = {"coordinate_file": autopick_df.loc[i]["rlnMicrographCoordinates"]}
        rlnInfo[absolut_path]["absolute_coordinate_file"] = project_file/ autopick_df.loc[i]["rlnMicrographCoordinates"]
        rlnInfo[absolut_path]["mrc_file"] = autopick_df.loc[i]["rlnMicrographName"]

        y,x = extract_coordinates(rlnInfo[absolut_path]["absolute_coordinate_file"])

        rlnInfo[absolut_path]["y"] = y
        rlnInfo[absolut_path]["x"] = x
        rlnInfo[absolut_path]["pickPath"] = pick_file
        files.append(absolut_path)


    if config["parameters"]["njobs"] <= 1:
        for file in tqdm.tqdm(files):
            rlnInfo[file]["mask"] = run_masking(file, config)
    else:
        with mp.get_context("spawn").Pool(config["parameters"]["njobs"]) as pool:
            results = [pool.apply_async(run_masking, args=[file, config]) for file in files]
            result = []
            for res in tqdm.tqdm(results):
                result.append(res.get())
            # result = [res.get() for res in result]
            for mask, file in zip(result, files):
                rlnInfo[file]["mask"] = mask
    
    exportRln(rlnInfo)


def exportRln(allRlnInfo):
    idxs = []

    # masks[img_data.rlnInfo["rlnMicrographCoordinates"]] = img_data.original_mask
    for absolut_path, rlnInfo in allRlnInfo.items():
        for idx, (y,x) in enumerate(zip(rlnInfo["y"],rlnInfo["x"])):
            if rlnInfo["mask"][y,x] == 1:
                idxs.append(idx)
        coordinate_file = rlnInfo["absolute_coordinate_file"]
        coordinate_df = starfile.read(coordinate_file)
        coordinate_df = coordinate_df.iloc[idxs]
        starfile.write(coordinate_df, rlnInfo["absolute_coordinate_file"],)


def get_recursiv_file(directory):
    new_files = glob.glob()
    


def get_files(config):
    input_files = config["command_line_arguments"]["input_files"]
    return_files = []
    if config["command_line_arguments"]["recursive"]:
        test_input_files = Path(input_files)
        if test_input_files.exists() and test_input_files.is_dir():
            to_iterate = [input_files]
        else:
            to_iterate = glob.glob(input_files)
        for file in to_iterate:
            file = Path(file)
            if file.is_dir():
                for dirpath, dirs, files in os.walk(file): 
                    for filename in files:
                        filename = Path(dirpath) / filename
                        if filename.suffix in suffix_function_dict:
                            return_files.append(filename)
            else:
                if file.suffix in suffix_function_dict:
                    return_files.append(file)
    else:
        for file in glob.glob(input_files):
            file = Path(file)
            if file.is_dir():
                continue
            if file.suffix in suffix_function_dict:
                return_files.append(file)

    return return_files


def run_masking(file, configs, return_for_csv=False):
    file = Path(file)
    mask = find_grid_hole_per_file(file, **configs["parameters"], pixel_size=configs["command_line_arguments"]["pixel_size"])
    data = None
    if configs["command_line_arguments"]["masked"]["save"]:
        output_dir = Path(configs["command_line_arguments"]["masked"]["output_dir"])
        data, ps = load_file(file)
        if ps is None:
            ps = configs["command_line_arguments"]["pixel_size"]
        
        if configs["command_line_arguments"]["masked"]["type"] == "mean":
            mask_value = np.mean(data[mask == 1])
        elif configs["command_line_arguments"]["masked"]["type"] == "min":
            mask_value = np.min(data[mask == 1])
        elif configs["command_line_arguments"]["masked"]["type"] == "max":
            mask_value = np.max(data[mask == 1])
        elif configs["command_line_arguments"]["masked"]["type"] == "random_noise":
            mean = np.mean(data[mask == 1])
            std = np.std(data[mask == 1])
            mask_value = np.random.normal(mean, std, size = (data[mask == 0]).size)
        else:
            print("\"type\" parameter for masked must be on of these:\nmean, min, max, random_noise")
            exit()
        data[mask == 0] = mask_value
        
        
        if configs["files"]["masked_file_type"] in [".mrc", ".MRC", ".rec", ".REC"]:
            output_path = output_dir / (file.stem + configs["files"]["masked_image_file_suffix"] + configs["files"]["masked_file_type"])
            
            with mrcfile.new(output_path, data=data, overwrite=False) as f:
                f.voxel_size = ps
        else:
            output_path = output_dir / (file.stem + configs["files"]["masked_image_file_suffix"] + configs["files"]["masked_file_type"])
            plt.imsave(output_path, data, cmap="gray")
        

    
    if configs["command_line_arguments"]["mask"]["save"]:
        output_dir = Path(configs["command_line_arguments"]["mask"]["output_dir"])        
        
        if configs["files"]["mask_file_type"] in [".mrc", ".MRC", ".rec", ".REC"]:
            _, ps = load_file(file)
            if ps is None:
                ps = configs["command_line_arguments"]["pixel_size"]
            output_path = output_dir / (file.stem + configs["files"]["mask_file_suffix"] + configs["files"]["mask_file_type"])
            with mrcfile.new(output_path, data=mask.astype(np.uint8), overwrite=False) as f:
                f.voxel_size = ps
        else:
            output_path = output_dir / (file.stem + configs["files"]["mask_file_suffix"] + configs["files"]["mask_file_type"])
            plt.imsave(output_path, mask, cmap="gray")


    if return_for_csv:
        if data is None:
            data, ps = load_file(file)
        mean = np.mean(data)
        median = np.median(data)
        fn = file
        found_edge = np.any(mask == 1)
        if not found_edge:
            percentage = 0
        else:
            percentage = np.sum(mask == 0) / np.array(mask).size
        return fn, found_edge, percentage, mean, median
    
       
    return mask

@run_ged.command()
@click.option("-c", "--config", default= Path.home() / ".config/GridEdgeDetector/config.toml", type=click.Path(True, True, False, readable=True,resolve_path=True))
def CLI(config):
    default_configs = load_config_file()
    configs = toml.load(config)
    
    default_configs.update(configs)

    to_run_cryosparc = default_configs["command_line_arguments"]["cryosparc"]["run"]
    to_run_relion = default_configs["command_line_arguments"]["relion"]["run"]
    default_configs["parameters"]["ring_width"] *= 10000
    default_configs["parameters"]["gridsizes"] = [i*10000 for i in default_configs["parameters"]["gridsizes"]]

    if to_run_cryosparc and to_run_relion:
        print("Config file says to run relion AND cryosparc. This is probably a mistake. If it is not, try to run twice with different settings. Will exit now.")
        exit()
    if to_run_cryosparc:
        run_cryosparc(default_configs)
    elif to_run_relion:
        run_relion(default_configs)
    else:
        print(default_configs)
        if len(default_configs["command_line_arguments"]["input_files"]) == 0:
            print("Parameter \"input_files\" should be a directory or files (simple wildcards possible).")
            exit()
        if not default_configs["command_line_arguments"]["masked"]["save"] and not default_configs["command_line_arguments"]["mask"]["save"] and not default_configs["command_line_arguments"]["csv"]["save"]:
            print("Either mask, masked or csv should be saved when not running cryosparc/relion jobs. Otherwise no output will be available.")
            exit()
    
        files = get_files(default_configs)

        csv_results = []
        if default_configs["parameters"]["njobs"] <= 1:
            for file in tqdm.tqdm(files) :
                csv_results.append(run_masking(file, default_configs, True))
        else:
            with mp.get_context("spawn").Pool(default_configs["parameters"]["njobs"]) as pool:
                results = [pool.apply_async(run_masking, args=[file, default_configs, True]) for file in files]
                # result = []
                for res in tqdm.tqdm(results):
                    csv_results.append(res.get())
                # result = [res.get() for res in result]
        if default_configs["command_line_arguments"]["csv"]["save"]:
            output_file = Path(default_configs["command_line_arguments"]["csv"]["output_file"])
            with open(output_file, "w") as f:
                f.write("file\tfound_edge\tpercentage_of_image_is_carbon\tmean\tmedian\n")
                argsorted_idx = np.argsort(np.array(csv_results)[...,2])
                for idx in argsorted_idx:
                    # f.write(f"{csv_results[idx][0]}\t{csv_results[idx][1]}\t{csv_results[idx][2]}\n")
                    f.write("\t".join([str(i) for i in csv_results[idx]]))
                    f.write("\n")



















if __name__ == "__main__":

    run_ged()