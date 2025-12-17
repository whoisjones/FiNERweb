# FiNERweb
A multilingual NER dataset covering 91 languages and 25 scripts. See our [paper](https://arxiv.org/abs/2512.13884) for details!

## Get Started

We host all materials on the [huggingface-hub](https://huggingface.co/collections/whoisjones/finerweb)!
The code for the project can be found [here](https://github.com/whoisjones/FiNERweb-code)!

**How to load the datasets**
```python
from datasets import load_dataset

finerweb = load_dataset('whoisjones/finerweb')
finerweb_de = load_dataset('whoisjones/finerweb', split='deu')
```

**How to load the regression models**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained("whoisjones/finerweb-multilabel-classifier-xlmr-4o")
    tokenizer = AutoTokenizer.from_pretrained("whoisjones/finerweb-multilabel-classifier-xlmr-4o")

    good_example = """'Kraft Foods has taken the Cadbury chocolate brand in a new direction, by combining it with cheese for the first time.
    The company is bringing together two of its brands and launching Philadelphia with Cadbury, a chilled chocolate spread made from Philadelphia Light and Cadbury chocolate.
    Kraft believes the new product has the potential to do very well and is targeting £10m in sales in the first year.
    The new cheese and chocolate spread is being launched on 1 February and will be appear in the chilled dairy aisle next to plain Philadelphia Light.
    It is launching in a 160g tub and a 120g four-pack of mini tubs, both with an rsp of £1.62.
    Kraft is supporting the launch with a £3.2m marketing budget in 2012 and is targeting 2,000 tonnes in volume sales – equivalent to about £10m – in the first year.
    If they reached this volume of sales, the new Philadelphia with Cadbury would have the same market value as Garlic & Herb, currently the biggest-selling flavour in the Philadelphia portfolio.
    Kraft already offers chocolate variants of Philadelphia in Italy and Germany, using Milka chocolate and targeting the breakfast occasion.
    In Germany, Philadelphia with Milka has generated €22.2m in sales since its October 2010 launch and has a 6.6% value share of the chocolate spread market.
    Kraft Foods UK marketing manager Bruce Newman said:
    “The UK product would be positioned as a snack.
    “The breakfast market in countries such as Germany is more developed, and our consumer research firmly identified Philadelphia with Cadbury as a snack.”'"""

    bad_example = """'|Viewing Single Post From: Spoilers for the Week of February 11th| |Lil||Feb 1 2013, 09:58 AM| Don\'t care about Chloe/Taniel/Jen-Jen . Don\'t care about Sami, really, but hoping that we get some good "SAMANTHA GENE!!" Marlena Death-Stares out of it . And "newfound" feelings . Please . If only . STEFANO!! STEFANO, STEFANO, STEFANO!!!!: cheer: |Spoilers for the Week of February 11th · DAYS: News, Spoilers & Discussion|'"""

    with torch.no_grad():
        good_example_inputs = tokenizer(good_example, return_tensors='pt')
        bad_example_inputs = tokenizer(bad_example, return_tensors="pt")
        good_example_outputs = model(**good_example_inputs)
        bad_example_outputs = model(**bad_example_inputs)
        print(good_example_outputs.logits)
        print(bad_example_outputs.logits)
```

## Datasets

- [FiNERweb](https://huggingface.co/datasets/whoisjones/fiNERweb)
- [FiNERweb-x](https://huggingface.co/datasets/whoisjones/fiNERweb-x) (translated labels)

## Regression Models

- [XLM-R (binary / 4o-mini)](https://huggingface.co/whoisjones/finerweb-binary-classifier-xlmr-4o)
- [XLM-R (multi-class / 4o-mini)](https://huggingface.co/whoisjones/finerweb-multilabel-classifier-xlmr-4o)
- [XLM-R (binary / Gemma3-27B)](https://huggingface.co/whoisjones/finerweb-binary-classifier-xlmr-gemma3)
- [XLM-R (multi-class / Gemma3-27B)](https://huggingface.co/whoisjones/finerweb-multilabel-classifier-mdeberta-gemma3)
- [mDeBERTa (binary / 4o-mini)](https://huggingface.co/whoisjones/finerweb-binary-classifier-mdeberta-4o)
- [mDeBERTa (multi-class / 4o-mini)](https://huggingface.co/whoisjones/finerweb-multilabel-classifier-mdeberta-4o)
- [mDeBERTa (binary / Gemma3-27B)](https://huggingface.co/whoisjones/finerweb-binary-classifier-mdeberta-gemma3)
- [mDeBERTa (multi-class / Gemma3-27B)](https://huggingface.co/whoisjones/finerweb-multilabel-classifier-xlmr-gemma3)

## Raw Materials
*Note*: These materials are the raw annotations, we recommend using the datasets above.

- [FiNERweb-gemma](https://huggingface.co/datasets/whoisjones/fiNERweb-gemma)
- [FiNERweb-4o](https://huggingface.co/datasets/whoisjones/fiNERweb-4o)
- [FiNERweb-multi](https://huggingface.co/datasets/whoisjones/fiNERweb-multi) (multilabel)

## Citation
If you find our work useful, please consider citing our [paper](https://arxiv.org/abs/2512.13884)!
```
@misc{golde2025finerwebdatasetsartifactsscalable,
      title={FiNERweb: Datasets and Artifacts for Scalable Multilingual Named Entity Recognition}, 
      author={Jonas Golde and Patrick Haller and Alan Akbik},
      year={2025},
      eprint={2512.13884},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.13884}, 
}
```
