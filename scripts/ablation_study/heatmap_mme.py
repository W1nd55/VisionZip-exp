import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("/Users/chenqinxinghao/Desktop/mme_summary_averages.csv")

df['alpha2_plus_alpha3'] = df['alpha2'] + df['alpha3']

plt.figure(figsize=(9, 6))
scatter = sns.scatterplot(
    x='alpha1', 
    y='mme_acc', 
    data=df,
    hue='alpha2_plus_alpha3',
    size='alpha2_plus_alpha3', 
    palette='viridis',
    sizes=(20, 200)
)

plt.title(r'mme_acc vs $\alpha_1$ colored by $(\alpha_2 + \alpha_3)$')
plt.xlabel(r'Visual Weight ($\alpha_1$)')
plt.ylabel('mme_acc')

max_acc = df['mme_acc'].max()
max_acc_alpha1 = df.loc[df['mme_acc'].idxmax(), 'alpha1']
plt.axhline(y=max_acc, color='r', linestyle='--', alpha=0.6, label='Overall Max ACC')
plt.text(0.1, max_acc - 0.001, f'Max ACC: {max_acc:.4f}', color='r')

plt.legend(title=r'Weight $\alpha_2+\alpha_3$', loc='lower right')
plt.grid(True, linestyle='--')
plt.show()