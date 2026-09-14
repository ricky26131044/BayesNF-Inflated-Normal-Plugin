# BayesNF Inflated-Normal Plugin (with Local Sigma)

傳統的貝氏神經場（Bayesian Neural Fields, BayesNF）在處理連續型時空資料時，預設的觀測模型通常為標準常態分配，且變異數為全域固定。然而，在處理特定領域的時空動態分析（例如：分析台灣各鄉鎮區域之肺癌死亡率）時，資料往往呈現兩大挑戰：
1. 因區域特性造成的極端「點膨脹」（Point-inflation）現象。
2. 不同區域的觀測值波動程度具有顯著差異（空間異質變異性）。

本擴充套件透過掛載全新的 `INFLATED_NORMAL` 觀測模型，賦予 BayesNF 處理這類極端分佈的能力。模型不僅會自動學習每個時空節點 $(s, t)$ 的膨脹機率與常態均值，更進一步引入了**局部標準差（Local Sigma）**的預測，大幅提升模型在複雜時空特徵下的擬合與預測表現。

---

##  核心特色 (Key Features)

** 無侵入式設計 (Non-invasive Patching)**  
  透過 `enable_mixture_normal()` 函數即可在 Runtime 動態攔截並替換 BayesNF 內部的 likelihood model，完全不需要修改原始碼。
  
** 三頭神經網路架構 (3-Head MLP Architecture)**  
  擴展原有的模型為分叉架構，建立共享主幹（深度為 depth-1 的層次）與三個獨立的輸出分支[cite: 2]。模型能同時預測常態分佈的均值（$F$）、點膨脹發生的 Logit 機率（$p$），以及局部噪聲對數尺度（$\log \sigma_{local}$）。
  
** 空間異質變異數建模 (Heteroscedastic Local Sigma)**  
  捨棄了全域單一的標準差設定，高斯成分的變異數 $\sigma$ 現在會隨著時空節點 $(s,t)$ 產生動態變化[cite: 2]。模型結合了全域基準偏置（Global baseline offset）與局部特徵調整，能更精準地捕捉區域間的波動差異。

** 高度彈性的點質量近似 (Point Mass Approximation)**  
  允許使用者自定義膨脹點位置 $c$（透過 `set_inflated_loc(v)`），並以極小變異（$\text{Scale} = 10^{-3}$）的常態分佈近似點質量。

---

##  數學模型 (Mathematical Formulation)

本套件將 likelihood 替換為具備局部變異數的混合模型。對於時空節點 $(s,t)$ 的觀測值 $y$，其分配定義如下：

$$y \sim p(s,t) \cdot \mathcal{N}(c, \sigma_{tiny}^2) + (1 - p(s,t)) \cdot \mathcal{N}(F(s,t), \sigma^2(s,t))$$

其中，局部標準差 $\sigma(s,t)$ 的計算方式為：
$$\sigma(s,t) = 0.01 + \exp(\text{log\_sigma\_bias} + \text{log\_sigma\_local}(s,t))$$

**參數說明：**
* $c$：使用者定義的膨脹點位置。
* $\sigma_{tiny}$：近似點質量的極小標準差（預設 $10^{-3}$）。
* $F(s,t)$：由第一輸出頭預測的觀測均值。
* $p(s,t)$：由第二輸出頭預測的膨脹發生機率（經 Sigmoid 轉換）。
* $\sigma(s,t)$：由第三輸出頭（搭配全域偏置）計算出的局部標準差。
* $\text{log\_sigma\_bias}$：全域基準偏置，控制 $\sigma$ 的整體量級。

---
