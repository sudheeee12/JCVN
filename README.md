# JCVN – Certificate Verification System (Blockchain + OCR)

JCVN is a blockchain-based certificate verification platform that allows users to:

- ✔️ Generate certificates  
- ✔️ Extract certificate data using OCR  
- ✔️ Compute Merkle root hashes  
- ✔️ Verify certificates using blockchain  
- ✔️ Prevent forgery & ensure authenticity  

This project uses **Python (Flask / Django)** + **Smart Contracts (Solidity)** + **Hardhat** + **OCR** to build a secure verification system.

---

## 🚀 Features

### ✅ 1. Certificate Generation
- Create digital certificate templates  
- Store templates locally  
- Export JSON certificate data  

### ✅ 2. OCR-Based Data Extraction
- Extract text from uploaded certificate images  
- Auto-fill certificate details for verification  

### ✅ 3. Merkle Root Hashing
- Generates a Merkle root for certificate data  
- Stores the root on blockchain  

### ✅ 4. Blockchain Verification
- Smart contract verifies certificate authenticity  
- Prevents tampering or fake modification  

### ✅ 5. Clean Frontend Templates
- Login page  
- Certificate build page  
- Verification page  
- OCR verification page  

---

## 🛠️ Tech Stack

### **Frontend**
- HTML  
- CSS  
- Jinja templates  

### **Backend**
- Python  
- Flask / Django  

### **Blockchain**
- Solidity  
- Hardhat  
- Ethers.js  
- Web3.js  

### **Other**
- OCR (Tesseract or similar)  
- JSON templates  
- Merkle Tree hashing  

---

## 📁 Project Structure
```
JCVN/
│── app.py
│── manage.py
│── .env
│── StoredMerkleRoot.json
│── stored_templates/
│── templates/
│   ├── base.html
│   ├── home.html
│   ├── login.html
│   ├── build-template.html
│   ├── verify.html
│   ├── verify-ocr.html
│   └── extract-data.html
│
└── blockchain/
    ├── contracts/
    │   └── MerkleRootHash.sol
    ├── scripts/
    │   └── deploy.js
    ├── hardhat.config.js
    ├── package.json
    └── package-lock.json
```

---

## ⚙️ Installation & Setup

### **1️⃣ Clone the Repository**
```
git clone https://github.com/sudheeee12/JCVN.git
cd JCVN
```

---
## **2️⃣ Install Python Requirements**
```
pip install -r requirements.txt
```

---
## **3️⃣ Setup Environment Variables**
Create a `.env` file (if not already present):
```
PRIVATE_KEY=your_metamask_private_key
RPC_URL=your_blockchain_network_url
CONTRACT_ADDRESS=deployed_contract_address
```
⚠️ Never share your private key publicly.

---
## **4️⃣ Install Blockchain Dependencies**
```
cd blockchain
npm install
```

---
## **5️⃣ Deploy Smart Contract**
```
npx hardhat run scripts/deploy.js --network sepolia
```
Copy the generated contract address into your `.env`.

---
## **6️⃣ Run the Application**
```
python app.py
```
The app will run at: **http://127.0.0.1:5000/**

---

## 🔍 How Verification Works
1. Certificate data is converted to a hash  
2. A Merkle tree is created from certificate fields  
3. The Merkle **root hash is stored on blockchain**  
4. During verification:  
   - OCR extracts data  
   - New Merkle root is generated  
   - Smart contract checks if it matches  

✔️ If matched → Certificate **Valid**  
❌ If not → **Tampered / Fake**

---








