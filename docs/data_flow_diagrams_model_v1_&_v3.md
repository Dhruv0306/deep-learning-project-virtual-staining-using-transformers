# Data Flow Diagrams

This document contains Mermaid data-flow diagrams for the Generator, Discriminator, and Training loops of **model_v1** and **model_v3**. Each component includes annotations for the tensor shapes passing between major operational blocks.

## Model V1

### 1. Generator (`ViTUNetGenerator`)
The v1 generator follows a U-Net backbone design with a pixel-wise Vision Transformer bottleneck.

```mermaid
graph TD
    %% Inputs
    X["Input Image (x)</br>Shape: (N, 3, 256, 256)"] --> E1
    
    subgraph Encoder ["Encoder"]
        E1["ConvBlock (e1)</br>Shape: (N, 64, 256, 256)"]
        D1["DownsampleBlock (d1)</br>Shape: (N, 128, 128, 128)"]
        E2["ConvBlock (e2)</br>Shape: (N, 128, 128, 128)"]
        D2["DownsampleBlock (d2)</br>Shape: (N, 256, 64, 64)"]
        E3["ConvBlock (e3)</br>Shape: (N, 256, 64, 64)"]
        D3["DownsampleBlock (d3)</br>Shape: (N, 512, 32, 32)"]
        E4["ConvBlock (e4)</br>Shape: (N, 512, 32, 32)"]
        D4["DownsampleBlock (d4)</br>Shape: (N, 512, 16, 16)"]
        
        E1 --> D1 --> E2 --> D2 --> E3 --> D3 --> E4 --> D4
    end

    subgraph Bottleneck ["Bottleneck"]
        B1["ConvBlock</br>Shape: (N, 512, 16, 16)"]
        FLAT["Flatten & Pos Embed</br>Shape: (N, 256, 512)"]
        VIT["PixelwiseViT Blocks (depth=4)</br>Shape: (N, 256, 512)"]
        UNFLAT["Unflatten</br>Shape: (N, 512, 16, 16)"]
        
        D4 --> B1 --> FLAT --> VIT --> UNFLAT
    end
    
    subgraph Decoder ["Decoder"]
        U1["UpsampleBlock (u1)</br>Shape: (N, 512, 32, 32)"]
        DEC1["Concat(u1, e4) -> ConvBlock (dec1)</br>Shape: (N, 512, 32, 32)"]
        
        U2["UpsampleBlock (u2)</br>Shape: (N, 256, 64, 64)"]
        DEC2["Concat(u2, e3) -> ConvBlock (dec2)</br>Shape: (N, 256, 64, 64)"]
        
        U3["UpsampleBlock (u3)</br>Shape: (N, 128, 128, 128)"]
        DEC3["Concat(u3, e2) -> ConvBlock (dec3)</br>Shape: (N, 128, 128, 128)"]
        
        U4["UpsampleBlock (u4)</br>Shape: (N, 64, 256, 256)"]
        DEC4["Concat(u4, e1) -> ConvBlock (dec4)</br>Shape: (N, 64, 256, 256)"]
        
        UNFLAT --> U1
        E4 -.-> DEC1
        U1 --> DEC1
        
        DEC1 --> U2
        E3 -.-> DEC2
        U2 --> DEC2
        
        DEC2 --> U3
        E2 -.-> DEC3
        U3 --> DEC3
        
        DEC3 --> U4
        E1 -.-> DEC4
        U4 --> DEC4
    end

    OUT["Output Conv (ReflectionPad + Conv2D + Tanh)</br>Shape: (N, 3, 256, 256)"]
    DEC4 --> OUT
```

### 2. Discriminator (`PatchDiscriminator`)
Standard PatchGAN identifying overlapping patches as real or fake.

```mermaid
graph TD
    X["Input Image (x)</br>Shape: (N, 3, 256, 256)"] --> L1
    
    L1["Conv2d (stride=2) + LReLU</br>Shape: (N, 64, 128, 128)"] --> L2
    L2["Conv2d (stride=2) + IN + LReLU</br>Shape: (N, 128, 64, 64)"] --> L3
    L3["Conv2d (stride=2) + IN + LReLU</br>Shape: (N, 256, 32, 32)"] --> L4
    L4["Conv2d (stride=1) + IN + LReLU</br>Shape: (N, 512, 31, 31)"] --> L5
    
    L5["Conv2d (Final Output)</br>Shape: (N, 1, 30, 30)"]
```

### 3. Training Loop
A standard unidirectional CycleGAN iteration with AMP and Gradient Scaling.

```mermaid
graph TD
    subgraph Data ["Data"]
        RA["Real A (Unstained)</br>Shape: (N, 3, 256, 256)"]
        RB["Real B (Stained)</br>Shape: (N, 3, 256, 256)"]
    end

    subgraph Generator_Step ["Generator Step (Freeze D)"]
        G_AB["G_AB Generator"]
        G_BA["G_BA Generator"]
        
        RA --> G_AB --> FB["Fake B</br>Shape: (N, 3, 256, 256)"]
        RB --> G_BA --> FA["Fake A</br>Shape: (N, 3, 256, 256)"]
        
        FB --> G_BA --> RecA["Reconstructed A</br>Shape: (N, 3, 256, 256)"]
        FA --> G_AB --> RecB["Reconstructed B</br>Shape: (N, 3, 256, 256)"]
        
        RA --> G_BA --> IdA["Identity A</br>Shape: (N, 3, 256, 256)"]
        RB --> G_AB --> IdB["Identity B</br>Shape: (N, 3, 256, 256)"]
        
        LossGen["Total G Loss</br>(LSGAN + Cycle + Identity + Perceptual)"]
        
        FB -.-> LossGen
        FA -.-> LossGen
        RecA -.-> LossGen
        RecB -.-> LossGen
        IdA -.-> LossGen
        IdB -.-> LossGen
    end

    subgraph Discriminator_Steps ["Discriminator Steps (Unfreeze D)"]
        DA["Discriminator A"]
        DB["Discriminator B"]
        
        RA --> DA
        FA -.-> DA
        DA --> LossDA["Loss D_A & Backward"]
        
        RB --> DB
        FB -.-> DB
        DB --> LossDB["Loss D_B & Backward"]
    end
    
    LossGen --> B_OPT_G["Update G_AB & G_BA</br>(Backward + Optimizer Step)"]
```

---

## Model V3

### 1. Generator (`CycleDiTGenerator` --> `DiTGenerator`)
A unified latent Diffusion Transformer (DiT) architecture driven by full and multi-scale conditioning tokens.

```mermaid
graph TD
    cond["Condition Image</br>Shape: (N, 3, 256, 256)"] --> CondT
    z_t["Noisy Latent (z_t)</br>Shape: (N, 4, 32, 32)"] --> PE
    domain["Target Domain ID</br>Shape: (N,)"] --> D_EMB
    t["Timestep (t)</br>Shape: (N,)"] --> T_EMB
    
    subgraph CondTokenizer ["Condition Tokenizer"]
        CondT["Multi-scale Proj (Full + 2x Pool)</br>Sum + Pos Embed"] --> CondTokens["Condition Tokens</br>Shape: (N, 256, 512)"]
    end
    
    D_EMB["Domain Embedding Layer"] --> D_T["Domain Token</br>Shape: (N, 512)"]
    D_T -. Added recursively via broadcast .-> CondTokens
    
    PE["PatchEmbed Stem (Overlapping Convs) + Pos Embed"] --> Tokens["Latent Tokens</br>Shape: (N, 256, 512)"]
    T_EMB["MLP(Sincos(t))"] --> T_D["Timestep Token</br>Shape: (N, 512)"]
    
    CondGlobal["Mean Pool of CondTokens</br>Shape: (N, 512)"]
    CondTokens --> CondGlobal
    T_D --> SUMCond["Global Modulator (cond)</br>Shape: (N, 512)"]
    CondGlobal --> SUMCond
    
    subgraph DiT_Blocks ["DiT Blocks (depth=8)"]
        Tokens --> Block1["DiT Block"]
        CondTokens --> Block1
        SUMCond --> Block1
        Block1 --> BlockN["... DiT Blocks ...</br>Alternating Full / Mean-Pooled Cross-Attention"]
    end
    
    BlockN --> TokensOutput["Tokens Output</br>Shape: (N, 256, 512)"]
    
    subgraph Linear_Head ["Linear Head & Unpatchify"]
        TokensOutput --> Lin["Linear Projection"] --> Unflat["Unpatchify & Reshape</br>Shape: (N, 256, 16) -> (N, 4, 32, 32)"]
    end
    
    Unflat --> VPred["Velocity/Noise Prediction (v_pred)</br>Shape: (N, 4, 32, 32)"]
```

### 2. Discriminator (`ProjectionDiscriminator`)
A composite 3-branch discriminator that addresses patches, global scale, and frequency domain. The output logits are scaled by learned normalisation weights.

```mermaid
graph TD
    X["Fake / Real Image</br>Shape: (N, 3, 256, 256)"] --> Branch1
    X --> Branch2
    X --> Branch3
    
    subgraph Branch_1 ["Branch 1: PatchGAN + MBStd"]
        Branch1["SpectralNormDiscriminator"] --> Feats1["Features</br>Shape: (N, c, 16, 16)"]
        Feats1 --> MB["MinibatchStdDev</br>(Appends 1 std feature channel)</br>Shape: (N, c+1, 16, 16)"]
        MB --> OUT1["Patch Logits</br>Shape: (N, 1, ~16, ~16)"]
    end
    
    subgraph Branch_2 ["Branch 2: Global Auto-Attention"]
        Branch2["CNN (Stride-4)"] --> Feats2["Shape: (N, 256, 4, 4)"]
        Feats2 --> Attn["Flatten to (N, 16, 256)</br>Self-Attention + LayerNorm"] 
        Attn --> H2["Reshape & Head</br>Shape: (N, 1, 1, 1)"]
        H2 --> OUT2["Scalar Logit</br>Shape: (N, 1)"]
    end
    
    subgraph Branch_3 ["Branch 3: Multi-channel FFT"]
        Branch3["Grayscale + R + G + B Extraction"]
        Branch3 --> FFT["Log1p(Magnitude(RFFT2))</br>Shape: (N, 4, H, W//2+1)"]
        FFT --> Norm["Sample Normalization"] --> CNN["CNN"] --> Pool["Global Mean Pool</br>Shape: (N, ~128)"]
        Pool --> OUT3["Scalar Logit</br>Shape: (N, 1)"]
    end
    
    OUT1 -.-> LogWeights["Softmax Learned Component Weights"]
    OUT2 -.-> LogWeights
    OUT3 -.-> LogWeights
    
    LogWeights --> FINAL["Output Weighted Tensors"]
```

### 3. Training Loop
The per-batch diffusion process separates the generator update into a diffusion state and an adversarial stage specifically to ease VRAM usage.

```mermaid
graph TD
    subgraph Initialize ["Initialize"]
        RA["Real A</br>Shape: (N, 3, 256, 256)"] --> EncodeA
        RB["Real B</br>Shape: (N, 3, 256, 256)"] --> EncodeB
        EncodeA["VAE Encode (Frozen)"] --> Z0A["Z_0 A</br>Shape: (N, 4, 32, 32)"]
        EncodeB["VAE Encode (Frozen)"] --> Z0B["Z_0 B</br>Shape: (N, 4, 32, 32)"]
    end

    subgraph Stage_1 ["Stage 1: Diffusion Generator Denoising Objective"]
        Z0A --> Add1["Add Noise (t_A)"] --> ZtA["Z_t A"]
        Z0B --> Add2["Add Noise (t_B)"] --> ZtB["Z_t B"]
        
        ZtA --> DiT_A["DiT Generator B2A"]
        ZtB --> DiT_B["DiT Generator A2B"]
        
        DiT_A --> DiffLossA["MSE/Perceptual Denoiser Loss"]
        DiT_B --> DiffLossB["MSE/Perceptual Denoiser Loss"]
        DiffLossA --> Back1["Backward Stage 1</br>(Activations freed)"]
        DiffLossB --> Back1
    end

    subgraph Stage_2 ["Stage 2: Adversarial & Cycle Objectives"]
        ZtA -.-> DiT_A2["Fresh Forward DiT A"] --> X0_A["x0_pred A</br>Shape: (N, 4, 32, 32)"]
        ZtB -.-> DiT_B2["Fresh Forward DiT B"] --> X0_B["x0_pred B</br>Shape: (N, 4, 32, 32)"]
        
        X0_A --> DecA["VAE Decode"] --> FB["Fake Image B</br>Shape: (N, 3, 256, 256)"]
        X0_B --> DecB["VAE Decode"] --> FA["Fake Image A</br>Shape: (N, 3, 256, 256)"]
        
        FB --> DB["D_B Evaluation"] --> AdvB["Adversarial Loss B"]
        FA --> DA["D_A Evaluation"] --> AdvA["Adversarial Loss A"]
        
        Z0A -.-> CycleLoss
        Z0B -.-> CycleLoss
        X0_A -.-> CycleLoss["Cycle & Identity Losses"]
        X0_B -.-> CycleLoss
        
        AdvA --> Back2["Backward Stage 2</br>& Step G"]
        AdvB --> Back2
        CycleLoss --> Back2
    end
    
    subgraph Discriminator_Steps ["Discriminator Steps"]
        D_A_real["D_A(Real A)"] 
        D_B_real["D_B(Real B)"] 
        
        FA_Buff["Fake A Buffer"] -.-> D_A_fake["D_A(Fake A)"]
        FB_Buff["Fake B Buffer"] -.-> D_B_fake["D_B(Fake B)"]
        
        D_A_real --> DLossA["D_A Loss & Step</br>(Including optional R1 penalty)"]
        D_A_fake --> DLossA
        
        D_B_real --> DLossB["D_B Loss & Step</br>(Including optional R1 penalty)"]
        D_B_fake --> DLossB
    end
```
