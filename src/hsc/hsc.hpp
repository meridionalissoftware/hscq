//  Copyright (c) 2026- David Lucius Severus
//
//  Distributed under the MIT License.
//  See accompanying file LICENSE or copy at
//  https://opensource.org/license/mit

#pragma once

//  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//  hsc/q
//  (Hopf spherical compression / quantization)
//  header-only, (micron corelib)
//  fixed-rate lossy VQ on spherical codes by Hopf foliations (Miyamoto/Costa/Sa Earp, IEEE T-IT 2021):
//  recursive S^(2n-1) decomposition to the 4D base case, no stored codebook, native comptime mode

#include "checksum.hpp"
#include "config.hpp"
#include "error.hpp"
#include "level.hpp"

#include "bits/bitreader.hpp"
#include "bits/bitwriter.hpp"

//  sphere math
#include "sphere/s2.hpp"
#include "sphere/s3.hpp"
#include "sphere/susp.hpp"
#include "sphere/tree.hpp"

#include "codec/block.hpp"
#include "codec/gain.hpp"
#include "codec/oct.hpp"
#include "codec/pack.hpp"
#include "codec/quat.hpp"
#include "codec/quotient.hpp"
#include "codec/rot.hpp"
#include "codec/scratch.hpp"

//  container and verbs
#include "format.hpp"
#include "hopf.hpp"
#include "unhopf.hpp"

//  fixed-rate size/ratio query (needs resolve + bound from hopf.hpp)
#include "rate.hpp"

//  porcelain: tensors, bits/weight targets, the preset ladder, the HSCQ container
#include "quant.hpp"

#include "ct.hpp"
