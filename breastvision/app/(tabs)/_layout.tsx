import { Tabs } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Platform } from "react-native";
import { HapticTab } from "@/components/haptic-tab";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { useColors } from "@/hooks/use-colors";

export default function TabLayout() {
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const bottomPadding = Platform.OS === "web" ? 10 : Math.max(insets.bottom, 8);
  return (
    <Tabs screenOptions={{ headerShown: false, tabBarActiveTintColor: "#146EF5", tabBarInactiveTintColor: "#7890A8", tabBarButton: HapticTab, tabBarStyle: { height: 58 + bottomPadding, paddingTop: 8, paddingBottom: bottomPadding, backgroundColor: colors.background, borderTopColor: "#D9E2EC", borderTopWidth: 0.5 } }}>
      <Tabs.Screen name="index" options={{ title: "Overview", tabBarIcon: ({ color }) => <IconSymbol name="house.fill" size={22} color={color} /> }} />
      <Tabs.Screen name="studies" options={{ title: "Studies", tabBarIcon: ({ color }) => <IconSymbol name="folder" size={22} color={color} /> }} />
      <Tabs.Screen name="assessment" options={{ title: "Assessment", tabBarIcon: ({ color }) => <IconSymbol name="analytics" size={22} color={color} /> }} />
      <Tabs.Screen name="robustness" options={{ title: "Robustness", tabBarIcon: ({ color }) => <IconSymbol name="show-chart" size={22} color={color} /> }} />
      <Tabs.Screen name="system" options={{ title: "System", tabBarIcon: ({ color }) => <IconSymbol name="settings" size={22} color={color} /> }} />
    </Tabs>
  );
}
