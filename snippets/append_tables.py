from os.path import expandvars
import argparse

import glob
import numpy as np

# PyTables
import tables as tb
# pandas data frames
import pandas as pd


def open_list_of_pytables_as_pandas(filename_list, destination):
    pyt_table = None
    outfile = tb.open_file(destination, mode="w")
    for i, filename in enumerate(sorted(filename_list)):

        if i == 0:
            pyt_infile = tb.open_file(filename, mode='r')
            pyt_table = pyt_infile.copy_node('/', name='reco_events',
                                             newparent=outfile.root)

        else:
            pyt_infile = tb.open_file(filename, mode='r')
            pyt_table_t = pyt_infile.root.reco_events
            pyt_table_t.append_where(dstTable=pyt_table)

    return pd.DataFrame(pyt_table[:])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--events_dir', type=str, default="data/events")
    parser.add_argument('--in_files_base', type=str, default="classified_events")
    args = parser.parse_args()

    for channel in ["gamma", "proton"]:
        for mode in ["tail", "wave"]:
            filename = "{}/{}_{}_{}_*.h5".format(args.events_dir, args.in_files_base,
                                                 channel, mode)
            open_list_of_pytables_as_pandas(glob.glob(filename),
                                            filename.replace("_*", ""))
