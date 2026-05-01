#include <bits/stdc++.h>
using namespace std;

unordered_map<string, unordered_set<string>> graph;
unordered_map<string, vector<int>> memo;
string cycle_start;

bool has_cycle(const string& current, int remaining) {
    auto it = memo.find(current);
    if (it != memo.end() && it->second[remaining] != -1)
        return it->second[remaining] == 1;

    if (memo.find(current) == memo.end())
        memo[current] = vector<int>(remaining + 1, -1);

    bool result = false;

    if (remaining == 0) {
        result = graph[current].count(cycle_start) > 0;
    } else {
        for (const string& next : graph[current]) {
            if (next == cycle_start) continue;
            if (has_cycle(next, remaining - 1)) {
                result = true;
                break;
            }
        }
    }

    memo[current][remaining] = result ? 1 : 0;
    return result;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int total_transactions = 0;
    long long total_money = 0;
    unordered_map<string, long long> money_from;
    set<string> accounts;

    // ── Data ingestion ──────────────────────────────────────
    string from, to, time_point, atm, token;
    long long money;

    while (cin >> token && token != "#") {
        from = token;
        cin >> to >> money >> time_point >> atm;

        total_transactions++;
        total_money += money;
        money_from[from] += money;
        graph[from].insert(to);
        accounts.insert(from);
        accounts.insert(to);
    }

    // ── Queries ─────────────────────────────────────────────
    string query;
    while (cin >> query && query != "#") {

        if (query == "?number_transactions") {
            cout << total_transactions << "\n";

        } else if (query == "?total_money_transaction") {
            cout << total_money << "\n";

        } else if (query == "?list_sorted_accounts") {
            bool first = true;
            for (const string& acc : accounts) {
                if (!first) cout << " ";
                cout << acc;
                first = false;
            }
            cout << "\n";

        } else if (query == "?total_money_transaction_from") {
            string account;
            cin >> account;
            cout << money_from[account] << "\n";

        } else if (query == "?inspect_cycle") {
            int k;
            cin >> cycle_start >> k;
            memo.clear();
            cout << (has_cycle(cycle_start, k - 1) ? 1 : 0) << "\n";
        }
    }

    return 0;
}