import wesep.models.tse_bsrnn_spk as bsrnn_spk
import wesep.models.tse_bsrnn_visual as bsrnn_visual

def get_model(model_name: str):
    if model_name.startswith("TSE_BSRNN_SPK"):
        return getattr(bsrnn_spk, model_name)
    elif model_name.startswith("TSE_BSRNN_VISUAL"):
        return getattr(bsrnn_visual, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
