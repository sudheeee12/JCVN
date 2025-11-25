const hre = require("hardhat");

async function main() {
  // Get the contract factory
  const StoreMerkleRoot = await hre.ethers.getContractFactory("StoreMerkleRoot");

  // Deploy contract (this already waits for deployment tx to be mined)
  const storeMerkleRoot = await StoreMerkleRoot.deploy();

  console.log(`StoreMerkleRoot deployed to: ${storeMerkleRoot.target}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

