def zeaz2xyz(az, ze):
    z = np.cos(ze)
    rz = np.sin(ze)
    x = rz*np.cos(az)
    y = rz*np.sin(az)
    return x, y, z

def xyz2zeaz(x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    #print('R', r)
    x = x/r
    y = y/r
    z = z/r
    ze = np.arccos(z)
    rz = np.sin(ze)
    az = np.arccos(x/rz)
    az = np.where(y < 0, np.pi*2-az, az)
    az = np.nan_to_num(az,0)
    return az, ze

PRED_DIR = '../work/'
META_DIR = '../input/icecube-neutrinos-in-deep-ice/train'
class MyDatasetFile(Dataset):
    def __init__(self, 
                 runNames, 
                 batch_ids,
                 tgt_cols = ['direction_x','direction_y','direction_z','direction_kappa', 'direction_x1','direction_y1','direction_z1','direction_kappa1'],
                 use_mid_fea = True,
                ):
        self.batch_ids = batch_ids
        self.runNames = runNames
        self.tgt_cols = tgt_cols
        self.use_mid_fea = use_mid_fea
        self.this_batch_id = -1
        self.this_batch_idx = -1
        self.reset_epoch()
        
    def reset_epoch(self) -> None:
        self.this_batch_idx += 1
        if self.this_batch_idx >= len(self.batch_ids):
            self.this_batch_idx = 0
            
        if self.this_batch_id == self.batch_ids[self.this_batch_idx]:
            print('skip reset epoch ', self.this_batch_id, self.this_batch_idx)
            return
        else:
            self.this_batch_id = self.batch_ids[self.this_batch_idx]
            print('reset epoch to batch_id:', self.this_batch_id, self.this_batch_idx)

        meta = pd.read_parquet(f'{META_DIR}/meta_{self.this_batch_id}.parquet').reset_index(drop=True)
        meta['pulse_count'] = np.log1p(meta.last_pulse_index - meta.first_pulse_index + 1)
        self.meta = meta
        
        result_list = []
        for runName in self.runNames:
            df = pd.read_pickle(f'{WORK_DIR}/{runName}_{self.this_batch_id}.pkl')
            if 'direction_kappa' in df:
                df['direction_kappa'] = np.log1p(df['direction_kappa'])
            if 'direction_kappa1' in df:
                df['direction_kappa1'] = np.log1p(df['direction_kappa1'])
            columns = self.tgt_cols
            if self.use_mid_fea:
                columns = columns + [f'idx{i}' for i in range(128)]
            result_list.append(df[columns].reset_index(drop=True))
        self.X = pd.concat(result_list, axis=1).values.astype('float32')

        self.event_ids = meta['event_id'].values
        if 'azimuth' in meta:
            #self.Y = np.stack(zeaz2xyz(results[0]['azimuth'], results[0]['zenith'])).T.astype('float32')
            self.Y = np.stack(zeaz2xyz(meta['azimuth'], meta['zenith'])).T.astype('float32')
            self.with_gt = True
        else:
            self.with_gt = False

    def __len__(self):
        return self.X.shape[0]
    
    def n_columns(self):
        return self.X.shape[1]

    def __getitem__(self, index):
        x = self.X[index]        
        event_id = self.event_ids[index]
        if self.with_gt:
            y = self.Y[index]
        else:
            y = None
        return x, y, event_id


TGT_COLS = ['direction_x','direction_y','direction_z','direction_kappa', 'direction_x1','direction_y1','direction_z1','direction_kappa1']
class MyDataset(Dataset):
    def __init__(self, results, meta, mode='train'):
        meta = meta.reset_index(drop=True)
        self.mode = mode
        result_list = []
        for result in results:
            if len(TGT_COLS):
                columns = TGT_COLS
            elif 'direction_x1' in result:
                print('use 8 fea')
                columns = ['direction_x','direction_y','direction_z','direction_kappa', 'direction_x1','direction_y1','direction_z1','direction_kappa1']
            else:
                columns = ['direction_x','direction_y','direction_z','direction_kappa']
            if USE_MID_FEA:
                columns = columns + [f'idx{i}' for i in range(128)]
            result_list.append(result[columns].reset_index(drop=True))
        self.X = pd.concat(result_list, axis=1).values.astype('float32')
        self.event_ids = results[0]['event_id'].values
        if 'azimuth' in meta:
            #self.Y = np.stack(zeaz2xyz(results[0]['azimuth'], results[0]['zenith'])).T.astype('float32')
            self.Y = np.stack(zeaz2xyz(meta['azimuth'], meta['zenith'])).T.astype('float32')
            self.with_gt = True
        else:
            self.with_gt = False

    def __len__(self):
        return self.X.shape[0]
    
    def n_columns(self):
        return self.X.shape[1]

    def __getitem__(self, index):
        x = self.X[index]
        event_id = self.event_ids[index]
        if self.mode == 'test':
            return x,0,event_id
        
        if self.with_gt:
            y = self.Y[index]
        else:
            y = None
        return x, y, event_id

class StandardModel3(Model):
    @save_model_config
    def __init__(
        self,
        *,
        tasks: Union[Task, List[Task]],
        n_input_fea, 
        dataset,
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__()

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, Task) for task in tasks)

        # Member variable(s)
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()
        self._n_input_fea = n_input_fea
        self._dataset = dataset
        
        mlp_layers = []
        layer_sizes = [n_input_fea, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE] # todo1
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            mlp_layers.append(torch.nn.Linear(nb_in, nb_out))
            mlp_layers.append(torch.nn.LeakyReLU())
            mlp_layers.append(torch.nn.Dropout(DROPOUT_PH2_MODEL))
        last_posting_layer_output_dim = nb_out

        self._mlp = torch.nn.Sequential(*mlp_layers)

            

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def forward(self, x):
        x = self._mlp(x)
        x = [task(x) for task in self._tasks]
        return x

    def training_step(self, xye, idx) -> Tensor:
        """Perform training step."""
        x,y,event_ids = xye
        preds = self(x)
        batch = Data(x=x, direction=y)
        vlosses = self._tasks[1].compute_loss(preds[1], batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], batch)
        tloss = torch.sum(tlosses)

        loss = vloss*0.1 + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss}

    def validation_step(self, xye, idx) -> Tensor:
        """Perform validation step."""
        x,y,event_ids = xye
        preds = self(x)
        batch = Data(x=x, direction=y)
        vlosses = self._tasks[1].compute_loss(preds[1], batch)
        vloss = torch.sum(vlosses)
        
        tlosses = self._tasks[0].compute_loss(preds[0], batch)
        tloss = torch.sum(tlosses)
        loss = vloss*0.1 + tloss
        return {"loss": loss, 'vloss': vloss, 'tloss': tloss}

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = None,
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        return super().predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )
    
    def training_epoch_end(self, training_step_outputs):
        loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in training_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in training_step_outputs]).mean()
        self.log_dict(
            {"trn_loss": loss, "trn_vloss": vloss, "trn_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, train loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')
        self._dataset.reset_epoch()
        
    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        vloss = torch.stack([x["vloss"] for x in validation_step_outputs]).mean()
        tloss = torch.stack([x["tloss"] for x in validation_step_outputs]).mean()
        self.log_dict(
            {"val_loss": loss, "val_vloss": vloss, "val_tloss": tloss},
            prog_bar=True,
            sync_dist=True,
        )
        print(f'epoch:{self.current_epoch}, valid loss:{loss.item()}, tloss:{tloss.item()}, vloss:{vloss.item()}')

def build_model3(config, dataloader, dataset) -> StandardModel2:
    """Builds GNN from config"""
    # Building model

    if config["target"] == 'direction':
        task = DirectionReconstructionWithKappa2(
            hidden_size=HIDDEN_SIZE,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
        )
        task2 = DirectionReconstructionWithKappa2(
            hidden_size=HIDDEN_SIZE,
            target_labels=config["target"],
            loss_function=DistanceLoss2(),
        )
        
        prediction_columns = [config["target"] + "_x", 
                              config["target"] + "_y", 
                              config["target"] + "_z", 
                              config["target"] + "_kappa" ]
        additional_attributes = ['zenith', 'azimuth', 'event_id']

    model = StandardModel3(
        tasks=[task2, task],
        n_input_fea=N_INPUT_FEA,
        dataset=dataset,
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        #optimizer_class=Lion,
        #optimizer_kwargs={"lr": 1e-04},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                10  * len(dataloader)//(len(config['gpus'])*config['accumulate_grad_batches'][0]),
                len(dataloader)*config["fit"]["max_epochs"]//(len(config['gpus'])*config['accumulate_grad_batches'][0]*2),
                len(dataloader)*config["fit"]["max_epochs"]//(len(config['gpus'])*config['accumulate_grad_batches'][0]),                
            ],
            "factors": [1e-03, 1, 1, 1e-03],
            "verbose": config["scheduler_verbose"],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes
    
    return model


def infer2(test_dataloader, model):
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
                pred = model(batch[0].to(f'cuda:{INF_DEVICES}'))
                #preds.append(pred[0])
                if USE_ALL_FEA_IN_PRED:
                    preds.append(torch.cat(pred, axis=-1))
                else:
                    preds.append(pred[0])
                event_ids.append(batch[2])
                if validateMode:
                    azimuth.append(batch[1][0])
                    zenith.append(batch[1][1])
        preds = torch.cat(preds).to('cpu').detach().numpy()
        #results = pd.DataFrame(preds, columns=model.prediction_columns)
        if USE_ALL_FEA_IN_PRED:
            if preds.shape[1] == 128+8:
                columns = ['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa'] + [f'idx{i}' for i in range(128)]
            else:
                columns = ['direction_x','direction_y','direction_z','direction_kappa'] + [f'idx{i}' for i in range(128)]
        else:
            if preds.shape[1] == 8:
                columns=['direction_x','direction_y','direction_z','direction_kappa1','direction_x1','direction_y1','direction_z1','direction_kappa']
            else:
                columns=['direction_x','direction_y','direction_z','direction_kappa']
        results = pd.DataFrame(preds, columns=columns)
        results['event_id'] = torch.cat(event_ids).to('cpu').detach().numpy()
            
        del zenith, azimuth, event_ids, preds
    else:
        results = model.predict_as_dataframe(
            gpus = config['gpus'],
            dataloader = test_dataloader,
            prediction_columns=model.prediction_columns,
            additional_attributes=model.additional_attributes,
        )
    gc.collect()
    return results


def calculate_angular_error2(df_pred, df_gt):
    df_gt = df_gt.reset_index(drop=True)
    df_pred = df_pred.reset_index(drop=True)
    df_pred['angular_error'] = np.arccos(df_gt['true_x']*df_pred['direction_x'] + df_gt['true_y']*df_pred['direction_y'] + df_gt['true_z']*df_pred['direction_z'])
    return df_pred

def convert_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Converts zenith and azimuth to 3D direction vectors"""
    df['true_x'] = np.cos(df['azimuth']) * np.sin(df['zenith'])
    df['true_y'] = np.sin(df['azimuth'])*np.sin(df['zenith'])
    df['true_z'] = np.cos(df['zenith'])
    return df

import time

HIDDEN_SIZE = 512
DROPOUT_PH2_MODEL = 0.0
N_INPUT_FEA = 136*len(result1_files)

USE_ALL_FEA_IN_PRED=False
INF_DEVICES = 0
USE_MID_FEA = True
CKPT = '../input/icecubedatas/base3-ens20-last.pth' # 0.9640
CKPT = '../input/icecubedatas/base4-ens23-last.pth' # 0.9637
CKPT = '../input/icecubedatas/base5-ens26-last.pth' # 0.9635
CKPT = '../input/icecubedatas/base5-ens32-6model-3layer-659batch-last.pth' # 0.963? todo1 

chunksize=50000
meta_chunks = pd.read_csv(CSV_META, chunksize=chunksize)
chunks = [pd.read_csv(f, chunksize=chunksize) for f in result1_files]
result_list = []

first_flg = True
while True:
    start_time = time.time()
    try:
        result_df = []
        for c in chunks:
            df = next(c)
            if 'direction_kappa' in df:
                df['direction_kappa'] = np.log1p(df['direction_kappa'])
            if 'direction_kappa1' in df:
                df['direction_kappa1'] = np.log1p(df['direction_kappa1'])
            result_df.append(df)
        this_meta = next(meta_chunks)
        this_meta['pulse_count'] = np.log1p(this_meta.last_pulse_index - this_meta.first_pulse_index + 1)
        test_dataset = MyDataset(result_df, this_meta, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
        if first_flg:
            model = build_model3(config = config, dataloader = test_dataloader, dataset = test_dataset)
            state_dict =  torch.load(CKPT, torch.device('cpu'))
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
            first_flg = False

        results2 = infer2(test_dataloader, model)
        if validateMode:
            this_meta = convert_to_3d(this_meta)
            results2 = calculate_angular_error2(results2, this_meta)
            print(results2.angular_error.mean())
        result_list.append(results2)
    except StopIteration:
        break
    del result_df
    gc.collect()
    print('total time', time.time()-start_time)
    
results2 = pd.concat(result_list)
if validateMode:
    print('all', results2.angular_error.mean())
    print('660', results2.angular_error[-153924:].mean())


def prepare_dataframe(df, angle_post_fix = '', vec_post_fix = '') -> pd.DataFrame:
    r = np.sqrt(df['direction_x'+ vec_post_fix]**2 + df['direction_y'+ vec_post_fix]**2 + df['direction_z' + vec_post_fix]**2)
    df['zenith' + angle_post_fix] = np.arccos(df['direction_z'+ vec_post_fix]/r)
    df['azimuth'+ angle_post_fix] = np.arctan2(df['direction_y'+ vec_post_fix],df['direction_x' + vec_post_fix]) #np.sign(results['true_y'])*np.arccos((results['true_x'])/(np.sqrt(results['true_x']**2 + results['true_y']**2)))
    df['azimuth'+ angle_post_fix][df['azimuth'  + angle_post_fix]<0] = df['azimuth'  + angle_post_fix][df['azimuth'  +  angle_post_fix]<0] + 2*np.pi 
    return df

results2 = prepare_dataframe(results2)
results2[['event_id','azimuth','zenith']].sort_values('event_id').to_csv("submission.csv", index = False)
results2[['event_id','azimuth','zenith']].sort_values('event_id')

