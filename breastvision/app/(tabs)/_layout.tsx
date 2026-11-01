import { Tabs } from "expo-router";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Platform, View, Text, Pressable, ActivityIndicator } from "react-native";
import { HapticTab } from "@/components/haptic-tab";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { useColors } from "@/hooks/use-colors";
import { useAuth } from "@/hooks/use-auth";
import * as Linking from "expo-linking";
import { OAUTH_PORTAL_URL, APP_ID } from "@/constants/oauth";

export default function TabLayout() {
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const { user, loading, logout, isAuthenticated } = useAuth();
  const bottomPadding = Platform.OS === "web" ? 10 : Math.max(insets.bottom, 8);

  const handleLogin = () => {
    if (Platform.OS === "web") {
      const redirectUri = window.location.origin + "/oauth/callback";
      const loginUrl = `${OAUTH_PORTAL_URL || "https://manus.im"}/oauth/authorize?client_id=${APP_ID}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code`;
      window.location.href = loginUrl;
    } else {
      const redirectUri = Linking.createURL("oauth/callback");
      const loginUrl = `${OAUTH_PORTAL_URL || "https://manus.im"}/oauth/authorize?client_id=${APP_ID}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code`;
      Linking.openURL(loginUrl);
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: colors.background }}>
      {/* Session Security Banner (OWASP Item 3 - Authentication Enforcement) */}
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          paddingHorizontal: 16,
          paddingTop: Math.max(insets.top, 8),
          paddingBottom: 8,
          backgroundColor: isAuthenticated ? "#F0FDF4" : "#FFFBEB",
          borderBottomWidth: 1,
          borderBottomColor: isAuthenticated ? "#BBF7D0" : "#FDE68A",
        }}
      >
        <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
          <View
            style={{
              width: 8,
              height: 8,
              borderRadius: 4,
              backgroundColor: isAuthenticated ? "#16A34A" : "#D97706",
            }}
          />
          <Text style={{ fontSize: 11, fontWeight: "700", color: isAuthenticated ? "#166534" : "#92400E" }}>
            {loading ? "Verifying clinical session..." : isAuthenticated ? `Logged in: ${user?.name || user?.email || "Clinician"}` : "Guest Demo Mode (Sign-in required for clinical EHR)"}
          </Text>
        </View>

        {loading ? (
          <ActivityIndicator size="small" color="#146EF5" />
        ) : isAuthenticated ? (
          <Pressable
            onPress={logout}
            style={{
              paddingVertical: 4,
              paddingHorizontal: 10,
              borderRadius: 6,
              backgroundColor: "#DC2626",
            }}
          >
            <Text style={{ color: "#FFFFFF", fontSize: 11, fontWeight: "700" }}>Log out</Text>
          </Pressable>
        ) : (
          <Pressable
            onPress={handleLogin}
            style={{
              paddingVertical: 4,
              paddingHorizontal: 10,
              borderRadius: 6,
              backgroundColor: "#146EF5",
            }}
          >
            <Text style={{ color: "#FFFFFF", fontSize: 11, fontWeight: "700" }}>Clinician Sign In</Text>
          </Pressable>
        )}
      </View>

      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarActiveTintColor: "#146EF5",
          tabBarInactiveTintColor: "#7890A8",
          tabBarButton: HapticTab,
          tabBarStyle: {
            height: 58 + bottomPadding,
            paddingTop: 8,
            paddingBottom: bottomPadding,
            backgroundColor: colors.background,
            borderTopColor: "#D9E2EC",
            borderTopWidth: 0.5,
          },
        }}
      >
        <Tabs.Screen
          name="index"
          options={{
            title: "Overview",
            tabBarIcon: ({ color }) => <IconSymbol name="house.fill" size={22} color={color} />,
          }}
        />
        <Tabs.Screen
          name="studies"
          options={{
            title: "Studies",
            tabBarIcon: ({ color }) => <IconSymbol name="folder" size={22} color={color} />,
          }}
        />
        <Tabs.Screen
          name="assessment"
          options={{
            title: "Assessment",
            tabBarIcon: ({ color }) => <IconSymbol name="analytics" size={22} color={color} />,
          }}
        />
        <Tabs.Screen
          name="robustness"
          options={{
            title: "Robustness",
            tabBarIcon: ({ color }) => <IconSymbol name="show-chart" size={22} color={color} />,
          }}
        />
        <Tabs.Screen
          name="system"
          options={{
            title: "System",
            tabBarIcon: ({ color }) => <IconSymbol name="settings" size={22} color={color} />,
          }}
        />
      </Tabs>
    </View>
  );
}
