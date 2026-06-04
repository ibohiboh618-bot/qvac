#include "AndroidDeviceInfo.hpp"

#ifdef __ANDROID__

#include <algorithm>
#include <cctype>

#include <sys/system_properties.h>

namespace qvac_lib_inference_addon_llama {
namespace android_device {

namespace {

std::string toLower(std::string_view str) {
  std::string result;
  result.reserve(str.size());
  for (char c : str) {
    result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return result;
}

std::optional<std::string> getProperty(const char* name) {
  char value[PROP_VALUE_MAX];
  if (__system_property_get(name, value) <= 0) {
    return std::nullopt;
  }
  return std::string(value);
}

bool startsWithIgnoreCase(std::string_view str, std::string_view prefix) {
  if (str.size() < prefix.size()) return false;
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(str[i])) != 
        std::tolower(static_cast<unsigned char>(prefix[i]))) {
      return false;
    }
  }
  return true;
}

} // namespace

std::optional<std::string> getManufacturer() {
  auto prop = getProperty("ro.product.manufacturer");
  return prop ? std::optional(toLower(prop.value())) : std::nullopt;
}

std::optional<std::string> getModel() {
  return getProperty("ro.product.model");
}

bool isSamsung() {
  auto mfg = getManufacturer();
  return mfg && startsWithIgnoreCase(mfg.value(), "samsung");
}

bool isS25Ultra() {
  auto model = getModel();
  return model && startsWithIgnoreCase(model.value(), "SM-S938");
}

bool isS26Ultra() {
  auto model = getModel();
  return  model && startsWithIgnoreCase(model.value(), "SM-S948");
}

bool isUltraDevice() {
  return isS25Ultra() || isS26Ultra();
}

} // namespace android_device
} // namespace qvac_lib_inference_addon_llama

#endif  // __ANDROID__
