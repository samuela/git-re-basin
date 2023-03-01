import matplotlib.pyplot as plt
import numpy as np

import matplotlib_style as _

# Results from Hyak runs by Jonathan.
weight_matching_results = [
    {
        'lambda': 0.0,
        'loss': 0.9565430283546448,
        'acc1': 0.7618799805641174,
        'acc5': 0.9297800064086914
    },
    {
        'lambda': 0.041666666666666664,
        'loss': 0.9514756202697754,
        'acc1': 0.7630000114440918,
        'acc5': 0.9294000267982483
    },
    {
        'lambda': 0.08333333333333333,
        'loss': 0.9564015865325928,
        'acc1': 0.7603999972343445,
        'acc5': 0.9279599785804749
    },
    {
        'lambda': 0.125,
        'loss': 0.9756367206573486,
        'acc1': 0.7542200088500977,
        'acc5': 0.925819993019104
    },
    {
        'lambda': 0.16666666666666666,
        'loss': 1.0161526203155518,
        'acc1': 0.7447199821472168,
        'acc5': 0.9202799797058105
    },
    {
        'lambda': 0.20833333333333331,
        'loss': 1.084486722946167,
        'acc1': 0.7278000116348267,
        'acc5': 0.9115800261497498
    },
    {
        'lambda': 0.25,
        'loss': 1.1932393312454224,
        'acc1': 0.7056999802589417,
        'acc5': 0.9003000259399414
    },
    {
        'lambda': 0.29166666666666663,
        'loss': 1.359868049621582,
        'acc1': 0.6745200157165527,
        'acc5': 0.8829799890518188
    },
    {
        'lambda': 0.3333333333333333,
        'loss': 1.5843234062194824,
        'acc1': 0.632420003414154,
        'acc5': 0.8587200045585632
    },
    {
        'lambda': 0.375,
        'loss': 1.8525607585906982,
        'acc1': 0.5848000049591064,
        'acc5': 0.8304799795150757
    },
    {
        'lambda': 0.41666666666666663,
        'loss': 2.113180160522461,
        'acc1': 0.5414800047874451,
        'acc5': 0.7993199825286865
    },
    {
        'lambda': 0.4583333333333333,
        'loss': 2.280402898788452,
        'acc1': 0.5138400197029114,
        'acc5': 0.7779600024223328
    },
    {
        'lambda': 0.5,
        'loss': 2.314577579498291,
        'acc1': 0.5101199746131897,
        'acc5': 0.7742400169372559
    },
    {
        'lambda': 0.5416666666666666,
        'loss': 2.194092035293579,
        'acc1': 0.5304399728775024,
        'acc5': 0.7902399897575378
    },
    {
        'lambda': 0.5833333333333333,
        'loss': 1.9772849082946777,
        'acc1': 0.569920003414154,
        'acc5': 0.8164399862289429
    },
    {
        'lambda': 0.625,
        'loss': 1.7209768295288086,
        'acc1': 0.615339994430542,
        'acc5': 0.8462799787521362
    },
    {
        'lambda': 0.6666666666666666,
        'loss': 1.4803653955459595,
        'acc1': 0.655780017375946,
        'acc5': 0.8715400099754333
    },
    {
        'lambda': 0.7083333333333333,
        'loss': 1.2947734594345093,
        'acc1': 0.6891800165176392,
        'acc5': 0.8912799954414368
    },
    {
        'lambda': 0.75,
        'loss': 1.1614160537719727,
        'acc1': 0.7139400243759155,
        'acc5': 0.9046199917793274
    },
    {
        'lambda': 0.7916666666666666,
        'loss': 1.0729618072509766,
        'acc1': 0.7307000160217285,
        'acc5': 0.912880003452301
    },
    {
        'lambda': 0.8333333333333333,
        'loss': 1.0164240598678589,
        'acc1': 0.7427200078964233,
        'acc5': 0.9185199737548828
    },
    {
        'lambda': 0.875,
        'loss': 0.9826194643974304,
        'acc1': 0.7498800158500671,
        'acc5': 0.9233400225639343
    },
    {
        'lambda': 0.9166666666666666,
        'loss': 0.9648454785346985,
        'acc1': 0.7551599740982056,
        'acc5': 0.9259799718856812
    },
    {
        'lambda': 0.9583333333333333,
        'loss': 0.9596028923988342,
        'acc1': 0.7583199739456177,
        'acc5': 0.927299976348877
    },
    {
        'lambda': 1.0,
        'loss': 0.9638407230377197,
        'acc1': 0.7585600018501282,
        'acc5': 0.9275199770927429
    },
]

otfusion_results = [{
    "lambda": 0.0,
    "acc1": 0.7619799971580505,
    "acc5": 0.9301400184631348
}, {
    "lambda": 0.0416666679084301,
    "acc1": 0.7616199851036072,
    "acc5": 0.9289600253105164
}, {
    "lambda": 0.0833333358168602,
    "acc1": 0.7581200003623962,
    "acc5": 0.9272800087928772
}, {
    "lambda": 0.125,
    "acc1": 0.7495800256729126,
    "acc5": 0.9236800074577332
}, {
    "lambda": 0.1666666716337204,
    "acc1": 0.7362200021743774,
    "acc5": 0.9159600138664246
}, {
    "lambda": 0.2083333432674408,
    "acc1": 0.7118399739265442,
    "acc5": 0.9031599760055542
}, {
    "lambda": 0.25,
    "acc1": 0.6706399917602539,
    "acc5": 0.8777199983596802
}, {
    "lambda": 0.2916666865348816,
    "acc1": 0.5908799767494202,
    "acc5": 0.8233199715614319
}, {
    "lambda": 0.3333333432674408,
    "acc1": 0.4489000141620636,
    "acc5": 0.7037000060081482
}, {
    "lambda": 0.375,
    "acc1": 0.2278199940919876,
    "acc5": 0.4514800012111664
}, {
    "lambda": 0.4166666865348816,
    "acc1": 0.06289999932050705,
    "acc5": 0.17047999799251556
}, {
    "lambda": 0.4583333432674408,
    "acc1": 0.017980000004172325,
    "acc5": 0.059379998594522476
}, {
    "lambda": 0.5,
    "acc1": 0.01372000016272068,
    "acc5": 0.0478999987244606
}, {
    "lambda": 0.5416666865348816,
    "acc1": 0.03200000151991844,
    "acc5": 0.09933999925851822
}, {
    "lambda": 0.5833333730697632,
    "acc1": 0.1325400024652481,
    "acc5": 0.2991600036621094
}, {
    "lambda": 0.625,
    "acc1": 0.34408000111579895,
    "acc5": 0.5936400294303894
}, {
    "lambda": 0.6666666269302368,
    "acc1": 0.5240799784660339,
    "acc5": 0.7721400260925293
}, {
    "lambda": 0.7083333134651184,
    "acc1": 0.631820023059845,
    "acc5": 0.8521400094032288
}, {
    "lambda": 0.75,
    "acc1": 0.6878200173377991,
    "acc5": 0.8891000151634216
}, {
    "lambda": 0.7916666865348816,
    "acc1": 0.72079998254776,
    "acc5": 0.9069200158119202
}, {
    "lambda": 0.8333333134651184,
    "acc1": 0.7375400066375732,
    "acc5": 0.9173799753189087
}, {
    "lambda": 0.875,
    "acc1": 0.7487800121307373,
    "acc5": 0.9226999878883362
}, {
    "lambda": 0.9166666865348816,
    "acc1": 0.7555599808692932,
    "acc5": 0.9256200194358826
}, {
    "lambda": 0.9583333134651184,
    "acc1": 0.7584400177001953,
    "acc5": 0.9273200035095215
}, {
    "lambda": 1.0,
    "acc1": 0.7587400078773499,
    "acc5": 0.9274799823760986
}]

naive_results = [
    {
        'lambda': 0.0,
        'loss': 0.9561377763748169,
        'acc1': 0.7622799873352051,
        'acc5': 0.9300000071525574
    },
    {
        'lambda': 0.041666666666666664,
        'loss': 0.9511223435401917,
        'acc1': 0.7611600160598755,
        'acc5': 0.9292600154876709
    },
    {
        'lambda': 0.08333333333333333,
        'loss': 0.966728687286377,
        'acc1': 0.7560999989509583,
        'acc5': 0.9277799725532532
    },
    {
        'lambda': 0.125,
        'loss': 1.015334129333496,
        'acc1': 0.745639979839325,
        'acc5': 0.9219599962234497
    },
    {
        'lambda': 0.16666666666666666,
        'loss': 1.1311142444610596,
        'acc1': 0.7249600291252136,
        'acc5': 0.9096199870109558
    },
    {
        'lambda': 0.20833333333333331,
        'loss': 1.3866240978240967,
        'acc1': 0.6886000037193298,
        'acc5': 0.8879799842834473
    },
    {
        'lambda': 0.25,
        'loss': 1.916377067565918,
        'acc1': 0.6225000023841858,
        'acc5': 0.8436999917030334
    },
    {
        'lambda': 0.29166666666666663,
        'loss': 2.9384732246398926,
        'acc1': 0.4955799877643585,
        'acc5': 0.7396199703216553
    },
    {
        'lambda': 0.3333333333333333,
        'loss': 4.4329328536987305,
        'acc1': 0.270440012216568,
        'acc5': 0.4952000081539154
    },
    {
        'lambda': 0.375,
        'loss': 5.873376369476318,
        'acc1': 0.059539999812841415,
        'acc5': 0.15651999413967133
    },
    {
        'lambda': 0.41666666666666663,
        'loss': 6.753904819488525,
        'acc1': 0.00687999976798892,
        'acc5': 0.02563999965786934
    },
    {
        'lambda': 0.4583333333333333,
        'loss': 7.080108165740967,
        'acc1': 0.00215999991632998,
        'acc5': 0.008320000022649765
    },
    {
        'lambda': 0.5,
        'loss': 7.143911838531494,
        'acc1': 0.0015399999683722854,
        'acc5': 0.007840000092983246
    },
    {
        'lambda': 0.5416666666666666,
        'loss': 6.995357513427734,
        'acc1': 0.0026199999265372753,
        'acc5': 0.01245999988168478
    },
    {
        'lambda': 0.5833333333333333,
        'loss': 6.403560161590576,
        'acc1': 0.019440000876784325,
        'acc5': 0.06261999905109406
    },
    {
        'lambda': 0.625,
        'loss': 5.22327995300293,
        'acc1': 0.14316000044345856,
        'acc5': 0.3098199963569641
    },
    {
        'lambda': 0.6666666666666666,
        'loss': 3.765409469604492,
        'acc1': 0.3731600046157837,
        'acc5': 0.6146600246429443
    },
    {
        'lambda': 0.7083333333333333,
        'loss': 2.545227527618408,
        'acc1': 0.5416399836540222,
        'acc5': 0.7800800204277039
    },
    {
        'lambda': 0.75,
        'loss': 1.778419017791748,
        'acc1': 0.6373199820518494,
        'acc5': 0.8533999919891357
    },
    {
        'lambda': 0.7916666666666666,
        'loss': 1.3582277297973633,
        'acc1': 0.6916599869728088,
        'acc5': 0.8890799880027771
    },
    {
        'lambda': 0.8333333333333333,
        'loss': 1.1438654661178589,
        'acc1': 0.7223600149154663,
        'acc5': 0.9092599749565125
    },
    {
        'lambda': 0.875,
        'loss': 1.033891201019287,
        'acc1': 0.739799976348877,
        'acc5': 0.9186400175094604
    },
    {
        'lambda': 0.9166666666666666,
        'loss': 0.9790284037590027,
        'acc1': 0.7518399953842163,
        'acc5': 0.924780011177063
    },
    {
        'lambda': 0.9583333333333333,
        'loss': 0.959520697593689,
        'acc1': 0.7569599747657776,
        'acc5': 0.9274799823760986
    },
    {
        'lambda': 1.0,
        'loss': 0.9631029367446899,
        'acc1': 0.7591400146484375,
        'acc5': 0.9274600148200989
    },
]

wm_lambdas = np.array([x["lambda"] for x in weight_matching_results])
otfusion_lambdas = np.array([x["lambda"] for x in otfusion_results])
naive_lambdas = np.array([x["lambda"] for x in naive_results])
np.testing.assert_allclose(wm_lambdas, otfusion_lambdas)
np.testing.assert_allclose(wm_lambdas, naive_lambdas)
lambdas = wm_lambdas

### Loss plot (without OT-Fusion)
fig = plt.figure()
ax = fig.add_subplot(111)

# Naive
ax.plot(
    lambdas,
    [x["loss"] for x in naive_results],
    color="grey",
    linewidth=2,
    linestyle="dashed",
)

# Weight matching
ax.plot(
    lambdas,
    [x["loss"] for x in weight_matching_results],
    color="tab:green",
    marker="^",
    linestyle="dashed",
    linewidth=2,
)

ax.plot([], [], color="grey", linewidth=2, label="Train")
ax.plot([], [], color="grey", linewidth=2, linestyle="dashed", label="Test")

ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$", "Model $B$"])
ax.set_ylabel("Loss")
ax.set_title(f"ImageNet, ResNet50 (1× width)")
ax.legend(loc="upper right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/imagenet_resnet50_loss_interp.png", dpi=300)
plt.savefig("figs/imagenet_resnet50_loss_interp.pdf")

# ### Loss plot (with OT-Fusion)
# fig = plt.figure()
# ax = fig.add_subplot(111)

# # Naive
# ax.plot(
#     lambdas,
#     [x["loss"] for x in naive_results],
#     color="grey",
#     linewidth=2,
#     linestyle="dashed",label="Naïve"
# )

# # Weight matching
# ax.plot(
#     lambdas,
#     [x["loss"] for x in weight_matching_results],
#     color="tab:green",
#     marker="^",
#     linestyle="dashed",
#     linewidth=2,label="Weight matching (ours)"
# )

# # OT-Fusion matching
# ax.plot(
#     lambdas,
#     [x["loss"] for x in otfusion_results],
#     color="tab:brown",
#     marker="x",
#     linestyle="dashed",
#     linewidth=2,label="OT-Fusion"
# )

# ax.set_xlabel("$\lambda$")
# ax.set_xticks([0, 1])
# ax.set_xticklabels(["Model $A$", "Model $B$"])
# ax.set_ylabel("Loss")
# ax.set_title(f"ImageNet, ResNet50 (1× width)")
# ax.legend(loc="upper right", framealpha=0.5)
# fig.tight_layout()

# plt.savefig("figs/imagenet_resnet50_loss_interp_with_otfusion.png", dpi=300)
# plt.savefig("figs/imagenet_resnet50_loss_interp_with_otfusion.pdf")

### Accuracy plot, top-1
fig = plt.figure()
ax = fig.add_subplot(111)

# Naive
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["train_acc_interp_naive"]),
#         color="grey",
#         linewidth=2,
#         label="Train")
ax.plot(
    lambdas,
    [100 * x["acc1"] for x in naive_results],
    color="grey",
    linewidth=2,
    linestyle="dashed",
    label="Test",
)

# Activation matching
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["train_acc_interp_clever"]),
#         color="tab:blue",
#         marker="*",
#         linewidth=2)
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["test_acc_interp_clever"]),
#         color="tab:blue",
#         marker="*",
#         linewidth=2,
#         linestyle="dashed")

# Weight matching
# ax.plot(lambdas, [x[""]], color="tab:green", marker="^", linewidth=2)
ax.plot(lambdas, [100 * x["acc1"] for x in weight_matching_results],
        color="tab:green",
        marker="^",
        linestyle="dashed",
        linewidth=2)

# STE matching
# ax.plot(lambdas,
#         np.array(ste_matching_run.summary["train_acc_interp_clever"]),
#         color="tab:red",
#         marker="p",
#         linewidth=2)
# ax.plot(lambdas,
#         np.array(ste_matching_run.summary["test_acc_interp_clever"]),
#         color="tab:red",
#         marker="p",
#         linestyle="dashed",
#         linewidth=2)

ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$", "Model $B$"])
ax.set_ylabel("Top-1 Accuracy")
ax.set_title(f"ImageNet, ResNet50 (1× width)")
# ax.legend(loc="lower right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/imagenet_resnet50_acc1_interp.png", dpi=300)
plt.savefig("figs/imagenet_resnet50_acc1_interp.pdf")

### Accuracy plot, top-5
fig = plt.figure()
ax = fig.add_subplot(111)

# Naive
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["train_acc_interp_naive"]),
#         color="grey",
#         linewidth=2,
#         label="Train")
ax.plot(
    lambdas,
    [100 * x["acc5"] for x in naive_results],
    color="grey",
    linewidth=2,
    linestyle="dashed",
    label="Test",
)

# Activation matching
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["train_acc_interp_clever"]),
#         color="tab:blue",
#         marker="*",
#         linewidth=2)
# ax.plot(lambdas,
#         np.array(activation_matching_run.summary["test_acc_interp_clever"]),
#         color="tab:blue",
#         marker="*",
#         linewidth=2,
#         linestyle="dashed")

# Weight matching
# ax.plot(lambdas, [x[""]], color="tab:green", marker="^", linewidth=2)
ax.plot(lambdas, [100 * x["acc5"] for x in weight_matching_results],
        color="tab:green",
        marker="^",
        linestyle="dashed",
        linewidth=2)

# STE matching
# ax.plot(lambdas,
#         np.array(ste_matching_run.summary["train_acc_interp_clever"]),
#         color="tab:red",
#         marker="p",
#         linewidth=2)
# ax.plot(lambdas,
#         np.array(ste_matching_run.summary["test_acc_interp_clever"]),
#         color="tab:red",
#         marker="p",
#         linestyle="dashed",
#         linewidth=2)

ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$", "Model $B$"])
ax.set_ylabel("Top-5 Accuracy")
ax.set_title(f"ImageNet, ResNet50 (1× width)")
# ax.legend(loc="lower right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/imagenet_resnet50_acc5_interp.png", dpi=300)
plt.savefig("figs/imagenet_resnet50_acc5_interp.pdf")

### Accuracy plot, top-1 (with OT-Fusion)
fig = plt.figure()
ax = fig.add_subplot(111)

# Naive
ax.plot(
    lambdas,
    [100 * x["acc1"] for x in naive_results],
    color="grey",
    linewidth=2,
    linestyle="dashed",
    label="Naïve",
)
ax.plot(lambdas, [100 * x["acc1"] for x in otfusion_results],
        color="tab:brown",
        marker="x",
        linestyle="dashed",
        linewidth=2,
        label="OT-Fusion")
ax.plot(lambdas, [100 * x["acc1"] for x in weight_matching_results],
        color="tab:green",
        marker="^",
        linestyle="dashed",
        linewidth=2,
        label="Weight matching (ours)")

ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$", "Model $B$"])
ax.set_ylabel("Top-1 Accuracy")
ax.set_title(f"ImageNet, ResNet50 (1× width)")
# ax.legend(loc="lower right", framealpha=0.5)
ax.legend(framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/imagenet_resnet50_acc1_interp_with_otfusion.png", dpi=300)
plt.savefig("figs/imagenet_resnet50_acc1_interp_with_otfusion.pdf")

### Accuracy plot, top-5 (with OT-Fusion)
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(
    lambdas,
    [100 * x["acc5"] for x in naive_results],
    color="grey",
    linewidth=2,
    linestyle="dashed",
    label="Naïve",
)
ax.plot(lambdas, [100 * x["acc5"] for x in otfusion_results],
        color="tab:brown",
        marker="x",
        linestyle="dashed",
        linewidth=2,
        label="OT-Fusion")
ax.plot(lambdas, [100 * x["acc5"] for x in weight_matching_results],
        color="tab:green",
        marker="^",
        linestyle="dashed",
        linewidth=2,
        label="Weight matching (ours)")

ax.set_xlabel("$\lambda$")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Model $A$", "Model $B$"])
ax.set_ylabel("Top-5 Accuracy")
ax.set_title(f"ImageNet, ResNet50 (1× width)")
# ax.legend(loc="lower right", framealpha=0.5)
fig.tight_layout()

plt.savefig("figs/imagenet_resnet50_acc5_interp_with_otfusion.png", dpi=300)
plt.savefig("figs/imagenet_resnet50_acc5_interp_with_otfusion.pdf")
