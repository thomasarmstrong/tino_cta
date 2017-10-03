#!/usr/bin/env python

from os.path import expandvars
import argparse

import glob

# PyTables
try:
    import tables as tb
except:
    raise
    pass
# pandas data frames
try:
    import pandas as pd
except:
    pass


def merge_list_of_pytables_as_pandas(filename_list, destination):
    pyt_table = None
    outfile = tb.open_file(destination, mode="w")
    for i, filename in enumerate(sorted(filename_list)):

        pyt_infile = tb.open_file(filename, mode='r')

        if i == 0:
            pyt_table = pyt_infile.copy_node('/reco_events/', name='table',
                                             newparent=outfile.root)

        else:
            pyt_table_t = pyt_infile.root.reco_events.table
            pyt_table_t.append_where(dstTable=pyt_table)

    return pd.DataFrame(pyt_table[:])


def merge_list_of_tables_as_pandas(filename_list, destination):
    store = pd.HDFStore(destination)
    for i, filename in enumerate(sorted(filename_list)):
        s = pd.HDFStore(filename)
        df = pd.read_hdf(filename, 'reco_events')
        if i == 0:
            store.put('reco_events', df, format='table', data_columns=True)
        else:
            store.append(key='reco_events', value=df, format='table')
    return store['reco_events']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--events_dir', type=str, default="data/prod3b/paranal/")
    parser.add_argument('--in_files_base', type=str, default="classified_events")
    parser.add_argument('--no_auto', action='store_false', dest='auto', default=True)
    parser.add_argument('--out_file', type=str)
    args = parser.parse_args()

    if args.auto:
        for channel in ["gamma", "proton"]:
            for mode in ["wave", "tail"]:
                filename = "{}/{}/{}_{}_{}_*.h5".format(
                        args.events_dir, mode,
                        args.in_files_base,
                        channel, mode)
                merge_list_of_tables_as_pandas(glob.glob(filename),
                                               filename.replace("_*", ""))
    else:
        merge_list_of_tables_as_pandas(
                glob.glob(args.events_dir+args.in_files_base+"*.h5"), args.out_file)
