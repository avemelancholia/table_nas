import torch
from torch import nn
from naslib.search_spaces.nasbenchasr.primitives import ASRPrimitive


class HeadOP(ASRPrimitive):
    def __init__(self, filters, num_classes):
        super().__init__(locals())
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=filters, out_features=num_classes)]
        )

    def forward(self, x, edge_data=None):
        output = self.layers[0](x)
        output = nn.functional.relu(output)
        return output


class TransformerOP(ASRPrimitive):
    def __init__(self, filters, dropout_rate=0):
        super().__init__(locals())
        self.tf_layer = nn.Transformer(
            d_model=filters,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=filters,
            dropout=dropout_rate,
            batch_first=True,
        )
        # self.tf_layer = nn.TransformerEncoderLayer(filters, 4, filters)

    def forward(self, x, edge_data=None):
        out = x.unsqueeze(1)
        out = self.tf_layer(out, out)
        out = out.mean(1)
        out = torch.clamp_max_(out, 20)
        # print(torch.mean(out))

        return out


class LinearOP(ASRPrimitive):
    def __init__(self, in_features, out_features, dropout_rate=0, name="Linear"):
        super().__init__(locals())
        self.name = name

        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_data=None):
        x = self.linear(x)
        x = self.relu(x)
        x = torch.clamp_max_(x, 20)
        x = self.dropout(x)
        return x

    def __repr__(self):
        return f"{self.__class__}({self.linear})"


class LinearBottleneckOP(ASRPrimitive):
    def __init__(self, filters, filters_wide, dropout_rate=0, name="Linear"):
        super().__init__(locals())
        self.name = name

        self.linear1 = nn.Linear(filters, filters_wide)
        self.relu1 = nn.ReLU(inplace=False)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(filters_wide, filters)
        self.relu2 = nn.ReLU(inplace=False)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_data=None):
        x = self.linear1(x)
        x = self.relu1(x)
        x = torch.clamp_max_(x, 20)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = torch.clamp_max_(x, 20)
        x = self.dropout2(x)
        return x

    def __repr__(self):
        return f"{self.__class__}({self.linear2})"


class ResBlockLinearOP(ASRPrimitive):
    def __init__(self, features, dropout_rate=0, name="resblock"):
        super().__init__(locals())
        self.name = name

        self.bn = nn.BatchNorm1d(features)
        self.linear1 = nn.Linear(features, features)
        self.relu = nn.ReLU(inplace=False)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.linear2 = nn.Linear(features, features)

    def forward(self, x, edge_data=None):
        out = self.bn(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = torch.clamp_max_(out, 20)

        out = self.dropout1(out)
        out = self.linear2(out)

        out = self.dropout2(out)

        return x + out

    def __repr__(self):
        return f"{self.__class__}({self.linear2})"
