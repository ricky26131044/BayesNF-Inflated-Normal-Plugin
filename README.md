# BayesNF Inflated-Normal Plugin (with Heteroscedastic Local Sigma)

傳統的貝氏神經場（Bayesian Neural Fields, BayesNF）在處理連續型時空資料時，通常預設觀測模型為單一常態誤差架構。然而，在處理特定領域的時空動態分析（例如：分析台灣各鄉鎮區域之肺癌死亡率）時，資料往往會因為區域特性或極端值而呈現嚴重的「點膨脹」（Point-inflation）現象。此外，不同鄉鎮在人口結構與醫療資源等條件上的差異，也使得各區域的隨機波動程度（變異數）具有顯著的空間異質性。

本擴充套件透過掛載全新的 `INFLATED_NORMAL` 觀測模型，將單一連續常態觀測模型擴展為可處理膨脹點與連續反應值並存的混合架構。模型在保留膨脹點訊息的同時，進一步刻畫非膨脹連續部分的時空變化與異質變異，大幅提升模型在複雜時空特徵下的預測能力。

---

##  核心特色 (Key Features)

1. **無侵入式設計 (Non-invasive Patching)**  
  透過 `enable_mixture_normal()` 函數即可在 Runtime 動態攔截並替換 BayesNF 內部的 likelihood model，完全不需要修改 BayesNF 的原始碼。
  
2. **三輸出頭神經網路架構 (3-Head MLP Architecture)**  
  擴展了原有的 `BayesianNeuralField1D` 模型，要求網路在每一個時空位置 $(s,t)$ 於最終隱藏層分叉輸出三個潛在時空函數 $(F, G, H)$。模型在一組時空特徵基礎上，能同時學習非膨脹部分期望值、膨脹部分機率值與異質性變異數三類訊息。
  
3. **具膨脹點與局部變異異質性之混合模型 (Mixture Model with Point Mass)**  
  模型假設觀測值有機率 *P(s,t)* 直接落在膨脹點（以集中於 *c* 之點質量分配 $δ_c$ 近似），而在其餘機率 *1 - P(s,t)* 下，觀測值則由以 *F(s,t)* 為平均結構、*σ²(s,t)* 為變異數之連續分布產生。

---

##  數學模型 (Mathematical Formulation)

本套件將 BayesNF 的觀測層建構為帶有膨脹點之混合模型。對於時空節點 $(s,t)$ 的觀測值 $Y(s,t)$，其機率結構定義如下：

$$Y(s,t) \vert{} P(s,t), F(s,t), \sigma(s,t) \sim P(s,t)\delta_c + (1 - P(s,t))\mathcal{N}(F(s,t), \sigma^2(s,t))$$

其中，混合模型的三個核心參數皆由神經網路的三個獨立輸出頭 $(F, G, H)$ 加上全域偏移項所構成：

1. **非膨脹連續部分之平均結構：** 
   直接由第一輸出頭 $F(s,t)$ 描述一般連續型觀測資料之潛在時空函數。
   
2. **膨脹機率結構 $P(s,t)$：** 
   由第二輸出頭 $G(s,t)$ 刻劃不同時空位置落於膨脹點之傾向，並加入全域偏移項 $b_{mixing}$ 調整整體基準：
   $$P(s,t) = \text{logit}^{-1}\{G(s,t) + b_{mixing}\}$$

3. **非膨脹連續部分之局部標準差 $\sigma(s,t)$：** 
   由第三輸出頭 $H(s,t)$ 描述對數尺度下的局部變異程度，並以 $b_\sigma$ 作為全域基準偏移：
   $$\sigma(s,t) = \epsilon_\sigma + \exp\{b_\sigma + H(s,t)\}$$


---
