import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from fvcore.common.config import CfgNode


datasets = ["covtype", "higgs-small", "otto", "adult", "churn"]


def get_numpys(root_dir, dataset):
    dss = {}
    for ds_type in ["train", "test"]:
        x_num = f"X_num_{ds_type}.npy"
        x_bin = f"X_bin_{ds_type}.npy"
        y = f"Y_{ds_type}.npy"

        x_bin = root_dir / dataset / x_bin
        x_num = root_dir / dataset / x_num

        if x_bin.exists() and x_num.exists():
            x_num = np.load(x_num)
            x_bin = np.load(x_bin)
            x = np.concatenate((x_num, x_bin), axis=1)
        elif x_bin.exists():
            x = np.load(x_bin)
        elif x_num.exists():
            x = np.load(x_num)
        else:
            raise NotImplementedError

        y = np.load(root_dir / dataset / y)

        dss[f"x_{ds_type}"] = x
        dss[f"y_{ds_type}"] = y
    return dss["x_train"], dss["y_train"], dss["x_test"], dss["y_test"]


def calculate_metrics(pred, gt):
    return {
        "accuracy": accuracy_score(gt, pred),
        "balanced_accuracy": balanced_accuracy_score(gt, pred),
    }


if __name__ == "__main__":
    with open("/home/table_nas/rf.yaml") as f:
        config = CfgNode.load_cfg(f)

    metrics = {}
    data_root = Path(config.data)
    for dataset in datasets:
        x_train, y_train, x_test, y_test = get_numpys(data_root, dataset)
        model = RandomForestClassifier(
            n_estimators=1000, max_depth=15, random_state=config.seed
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics[dataset] = calculate_metrics(y_pred, y_test)
        print(dataset)
        print(metrics[dataset])

    out_dir = Path(config.save)

    with open("/home/experiments/random_forest.pickle", "wb") as outfile:
        pickle.dump(metrics, outfile)
