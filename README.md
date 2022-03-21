# 🦖 BiTe-REx: **Bi**lingual **Te**xt **R**etrieval **Ex**planations<br>in the Automotive domain


Find the **demo application** here: [BiTe-REx](https://bite-rex-demo.herokuapp.com/).

To satiate the comprehensive information need of users, retrieval systems surpassing the boundaries of language are inevitable in the present digital space in the wake of an ever-rising multilingualism. This work presents the first-of-its kind **Bi**lingual **Te**xt **R**etrieval **Ex**planations (BiTe-REx) aimed at users performing competitor or wage analysis in the automotive domain. BiTe-REx supports users to gather a more comprehensive picture of their query by retrieving results regardless of the query language and enables them to make a more informed decision by exposing how the underlying model judges the relevance of documents. With a user study, we demonstrate statistically significant results on the understandability and helpfulness of the explanations provided by the system.

<p align="center">
<img src="data/figures/overview.png" alt="drawing" style="width:600px;"/>
</p>

### How to run the app?

You can run the **demo app** using the link [BiTe-REx](https://bite-rex-demo.herokuapp.com/) or using the following commands in your local machine.

```
$python3.8.8
pip install -r requirements.txt
streamlit run src/bite_rex_demo.py
```

You can run the **full app** using the following commands in your local machine.

```
$python3.8.8
pip install -r requirements_full.txt
streamlit run src/bite_rex.py
```

