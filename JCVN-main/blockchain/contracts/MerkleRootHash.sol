// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StoreMerkleRoot {
    // Array to store Merkle root hashes
    bytes32[] private rootHashes;
    mapping(bytes32 => bool) private rootExists;

    // Store a single Merkle root
    function storeRoot(bytes32 _rootHash) external {
        require(!rootExists[_rootHash], "Root already exists");
        rootHashes.push(_rootHash);
        rootExists[_rootHash] = true;
    }

    // Store multiple Merkle roots in one transaction
    function storeRoots(bytes32[] calldata _rootHashes) external {
        for (uint256 i = 0; i < _rootHashes.length; i++) {
            if (!rootExists[_rootHashes[i]]) {
                rootHashes.push(_rootHashes[i]);
                rootExists[_rootHashes[i]] = true;
            }
        }
    }

    // Get total number of stored roots
    function getRootCount() external view returns (uint256) {
        return rootHashes.length;
    }

    // Check if a root exists
    function hasRoot(bytes32 _rootHash) external view returns (bool) {
        return rootExists[_rootHash];
    }
}

