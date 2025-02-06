validateMode = False
USE_ALL_FEA_IN_PRED=True
INF_DEVICES = 0
result1_files = []

def is_env_notebook():
    """Determine wheather is the environment Jupyter Notebook"""
    if 'get_ipython' not in globals():
        # Python shell
        return False
    env_name = get_ipython().__class__.__name__
    if env_name == 'TerminalInteractiveShell':
        # IPython shell
        return False
    # Jupyter Notebook
    return True


def convert_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Converts zenith and azimuth to 3D direction vectors"""
    df['true_x'] = np.cos(df['azimuth']) * np.sin(df['zenith'])
    df['true_y'] = np.sin(df['azimuth'])*np.sin(df['zenith'])
    df['true_z'] = np.cos(df['zenith'])
    return df

def calculate_angular_error(df : pd.DataFrame) -> pd.DataFrame:
    """Calcualtes the opening angle (angular error) between true and reconstructed direction vectors"""
    df['angular_error'] = np.arccos(df['true_x']*df['direction_x'] + df['true_y']*df['direction_y'] + df['true_z']*df['direction_z'])
    return df

def infer(min_pulse, max_pulse, batch_size, this_batch_id):
    if validateMode:
        labels = {'direction': Direction()}
    else:
        labels = None
    print('labels', labels)
    test_dataloader, test_dataset = make_dataloader2(db = "dummy",
                                                selection = None,
                                                pulsemaps = config['pulsemap'],
                                                features = features,
                                                truth = truth,
                                                batch_ids = [this_batch_id],
                                                batch_size = batch_size,
                                                num_workers = config['num_workers'],
                                                shuffle = False,
                                                labels = labels,
                                                index_column = config['index_column'],
                                                truth_table = config['truth_table'],
                                                max_len = 0,
                                                max_pulse = max_pulse,
                                                min_pulse = min_pulse,
                                                )

    if len(test_dataset) == 0:
        print('skip inference')
        return pd.DataFrame()
    
    model = build_model2(config, test_dataloader, test_dataset)

    state_dict =  torch.load(CKPT, torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    USE_ORIG_PRED = True
    if USE_ORIG_PRED:
        event_ids = []
        zenith = []
        azimuth = []
        preds = []
        print('start predict')
        with torch.no_grad():
            model.eval()
            model.to(f'cuda:{INF_DEVICES}')
            for batch in tqdm(test_dataloader):
                pred = model(batch.to(f'cuda:{INF_DEVICES}'))
                #preds.append(pred[0])
                if USE_ALL_FEA_IN_PRED:
                    preds.append(torch.cat(pred, axis=-1))
                else:
                    preds.append(pred[0])
                event_ids.append(batch.event_id)
                if validateMode:
                    zenith.append(batch.zenith)
                    azimuth.append(batch.azimuth)
        preds = torch.cat(preds).to('cpu').detach().numpy()
        #results = pd.DataFrame(preds, columns=model.prediction_columns)
        if USE_ALL_FEA_IN_PRED:
            if preds.shape[1] == 128+8:
                columns = ['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa'] + [f'idx{i}' for i in range(128)]
            else:
                columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
        else:
            columns=model.prediction_columns
        results = pd.DataFrame(preds, columns=columns)
        results['event_id'] = torch.cat(event_ids).to('cpu').detach().numpy()
        if validateMode:
            results['zenith'] = torch.cat(zenith).to('cpu').numpy()
            results['azimuth'] = torch.cat(azimuth).to('cpu').numpy()
            
        del zenith, azimuth, event_ids, preds
    else:
        results = model.predict_as_dataframe(
            gpus = config['gpus'],
            dataloader = test_dataloader,
            prediction_columns=model.prediction_columns,
            additional_attributes=model.additional_attributes,
        )
    gc.collect()
    if validateMode:
        results = convert_to_3d(results)
        results = calculate_angular_error(results)
        print('angular_error',results["angular_error"].mean())
    return results


if validateMode:
    BATCH_DIR = '../input/icecube-neutrinos-in-deep-ice/train'
    #meta = pd.read_parquet('../input/icecubedatas/train_meta_651to660.parquet')
    #META_DIR = '../work/test_valid10'
    meta = pd.read_parquet('../input/icecubedatas/train_meta_656to660.parquet')
    META_DIR = '../work/test_valid5'
    #meta = pd.read_parquet('../input/icecubedatas/train_meta_660.parquet')
    #META_DIR = '../work/test_valid1'
else:
    BATCH_DIR = '../input/icecube-neutrinos-in-deep-ice/test'
    META_DIR = '../work/test'
    meta = pd.read_parquet('../input/icecube-neutrinos-in-deep-ice/test_meta.parquet')
    
WORK_DIR = META_DIR
os.makedirs(META_DIR, exist_ok = True)
CSV_META = f'{META_DIR}/test_meta.csv'
if not os.path.exists(CSV_META):
    meta.to_csv(CSV_META, index=False)
batch_ids = meta.batch_id.unique()
for batch_id in batch_ids:
    out_file = f'{META_DIR}/meta_{batch_id}.parquet'
    if os.path.exists(out_file):
        continue
    meta[meta.batch_id == batch_id].reset_index(drop=False).to_parquet(out_file, index=False)
del meta
_ = gc.collect()

ONLY_AUX_FALSE = False
runName = '4l-ph3'
CKPT = '../input/icecubedatas/base1-4l-lr2-batch1000-splitModel2-650x14-retryFromStart2-last.pth'
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,3)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '4l-retry1-e2t10-ph3'
CKPT = '../input/icecubedatas/base1-4l-splitModel2-650x2-theta10-retryFromStart3-last.pth'
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,3)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '3lnoPP-ph3'
CKPT = '../input/icecubedatas/base1-3l300p500b-noPP-650x2-retryFromStart2-epoch1299-val_tloss0.999595.ckpt' # 0.9689275622367859, 0.9656946659088135
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,3)]
USE_G = False
USE_PP = False
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '3l4n-avg'
CKPT = '../input/icecubedatas/base1-3l250p4n-batch1000-650x8-retryFromStart1-avg.pth' # 0.9705609679222107
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,4)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '4l-splitModel2-650x12-ph3'
CKPT = '../input/icecubedatas/base1-4l-splitModel2-650x12-retryFromStart4-retry1-last.pth' #0.9661051034927368
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,3)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

ONLY_AUX_FALSE = False
runName = '4l4D500p-batch500-splitModel2-650x20-ph3'
CKPT = '../input/icecubedatas/base1-4l4D500p-batch500-splitModel2-650x20-retry4-last.pth' # 0.9676517248153687
COLUMNS_NEAREST_NEIGHBOURS = [slice(0,4)]
USE_G = True
USE_PP = True
DYNEDGE_LAYER_SIZE = [
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
                (
                    256,
                    256,
                ),
            ]

results_all_batch = []
for this_batch_id in batch_ids: #TODO
    result0 = infer(0, 96, 400, this_batch_id); gc.collect()
    result1 = infer(96, 140, 200, this_batch_id); gc.collect()
    result2 = infer(140, 300, 100, this_batch_id); gc.collect()
    result3 = infer(300, 1000, 20, this_batch_id); gc.collect()
    result4 = infer(1000, 2000, 2, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    result5 = infer(2000, 0, 1, this_batch_id); gc.collect() # left all with FORCE_MAX_PULSE
    results = pd.concat([result0, result1, result2, result3, result4, result5]).sort_values('event_id')
    results_all_batch.append(results)

    if validateMode:
        print('angular_error',this_batch_id, results["angular_error"].mean())

    del result0, result1, result2, result3, result4, result5, results
    gc.collect()

results_all_batch = pd.concat(results_all_batch).sort_values('event_id').reset_index(drop=True)
results_all_batch['event_id'] = results_all_batch['event_id'].astype(int)
results_all_batch.to_csv(f'{WORK_DIR}/{runName}_{validateMode}.csv', index=False)
result1_files.append(f'{WORK_DIR}/{runName}_{validateMode}.csv')

if validateMode:
    print('angular_error',batch_ids, results_all_batch["angular_error"].mean())

del results_all_batch
gc.collect()

