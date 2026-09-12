// Fallback for using MaterialIcons on Android and web.

import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import { SymbolWeight } from "expo-symbols";
import { ComponentProps } from "react";
import { OpaqueColorValue, type StyleProp, type TextStyle } from "react-native";

type IconMapping = Record<string, ComponentProps<typeof MaterialIcons>["name"]>;
type IconSymbolName = keyof typeof MAPPING;

/**
 * Add your SF Symbols to Material Icons mappings here.
 * - see Material Icons in the [Icons Directory](https://icons.expo.fyi).
 * - see SF Symbols in the [SF Symbols](https://developer.apple.com/sf-symbols/) app.
 */
const MAPPING = {
  "house.fill": "home",
  "paperplane.fill": "send",
  "chevron.left.forwardslash.chevron.right": "code",
  "chevron.right": "chevron-right",
  "waveform.path.ecg": "monitor-heart",
  "plus": "add",
  "check-circle": "check-circle",
  "error": "error",
  "priority-high": "priority-high",
  "info": "info",
  "visibility-off": "visibility-off",
  "science": "science",
  "add-photo-alternate": "add-photo-alternate",
  "image-search": "image-search",
  "bar-chart": "bar-chart",
  "close": "close",
  "crop-free": "crop-free",
  "cloud-upload": "cloud-upload",
  "play-arrow": "play-arrow",
  "expand-more": "expand-more",
  "folder": "folder",
  "analytics": "analytics",
  "settings": "settings",
  "search": "search",
  "folder-open": "folder-open",
  "delete-outline": "delete-outline",
  "lock": "lock",
  "sync": "sync",
  "radio-button-unchecked": "radio-button-unchecked",
  "show-chart": "show-chart",
} as IconMapping;

/**
 * An icon component that uses native SF Symbols on iOS, and Material Icons on Android and web.
 * This ensures a consistent look across platforms, and optimal resource usage.
 * Icon `name`s are based on SF Symbols and require manual mapping to Material Icons.
 */
export function IconSymbol({
  name,
  size = 24,
  color,
  style,
}: {
  name: IconSymbolName;
  size?: number;
  color: string | OpaqueColorValue;
  style?: StyleProp<TextStyle>;
  weight?: SymbolWeight;
}) {
  return <MaterialIcons color={color} size={size} name={MAPPING[name]} style={style} />;
}
