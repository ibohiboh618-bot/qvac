#pragma once

#include <optional>
#include <string>

namespace qvac_lib_inference_addon_llama {

#ifdef __ANDROID__

namespace android_device {

/// Get device manufacturer (lowercased)
std::optional<std::string> getManufacturer();

/// Get device model (as-is from system property)
std::optional<std::string> getModel();

/// Check if manufacturer is Samsung (case-insensitive)
bool isSamsung();

/// Check if model starts with "SM-S938" (S25 Ultra variants)
bool isS25Ultra();

/// Check if model starts with "SM-S948" (S26 Ultra variants)
bool isS26Ultra();

/// Check if device is S25 Ultra OR S26 Ultra
bool isUltraDevice();

} // namespace android_device

#endif // __ANDROID__

} // namespace qvac_lib_inference_addon_llama
