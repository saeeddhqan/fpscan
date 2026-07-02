#pragma once

#include <cstdint>
#include <vector_types.h>

template <int Bytes>
struct BytesToType;

template <> struct BytesToType<16> { using Type = uint4; };
template <> struct BytesToType<8>  { using Type = uint2; };
template <> struct BytesToType<4>  { using Type = uint32_t; };
template <> struct BytesToType<2>  { using Type = uint16_t; };
template <> struct BytesToType<1>  { using Type = uint8_t; };
