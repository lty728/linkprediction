#include <vector>
#include <ctime>
#include <iostream>
#include <random>
#include <fstream>  
#include <string>  
#include <sstream>
#include <cmath>
#include <queue>

using namespace std;

class Graph {
private:
	int nsize;//节点数量
	int M;//边数
	int testnum;//测试边数
	int realsize;
	vector<vector<int> > n; //二维向量存储每个节点的邻接节点
	vector<vector<int> > tempn;//用于生成不存在边
	vector<vector<int> > test;//测试数据
	double AUC;
	queue<int> q; //搜索队列
	vector<bool> visited; //访问标志
	vector<int> level;//广度优先遍历层数

	default_random_engine e;//随机数引擎


public:

	void readdata(string filename) {
		ifstream in(filename);
		string line;
		if (in) // 有该文件  
		{
			int n1, n2;
			n.clear();
			tempn.clear();
			nsize = 0;
			M = 0;
			while (getline(in, line)) // line中不包括每行的换行符  
			{
				M++;
				istringstream is(line);
				is >> n1 >> n2;
				if ((n1 >= nsize) || (n2 >= nsize)) {
					nsize = ((n1 > n2) ? n1 : n2) + 1;
					n.resize(nsize);
				}
				n[n1].push_back(n2);
				n[n2].push_back(n1);
			}
			tempn.assign(n.begin(), n.end());
			realsize = 0;
			for (int i = 0; i < nsize; i++) {
				if (n[i].size() != 0) {
					realsize++;
				}
				else
				{
					cout << i << endl;
				}
			}
			e.seed(time(0));
		}
		else // 没有该文件  
		{
			cout << "no such file" << endl;
			return;
		}
		cout << "data loaded" << endl;
	}


	bool connected(int node1, int node2) {
		vector<int> visited;
		visited.resize(nsize, false);
		visited[node1] = true;
		queue<int> q;
		for (int i = 0; i < n[node1].size(); i++) {
			if (n[node1][i] != node2) {
				q.push(n[node1][i]);
				visited[n[node1][i]] = true;
			}
		}
		while (!q.empty()) {
			int v = q.front();
			q.pop();
			for (int i = 0; i < n[v].size(); i++) {
				int now = n[v][i];
				if (!visited[now]) {
					if (now == node2) {
						return true;
					}
					q.push(now);
					visited[now] = true;
				}
			}
		}
		return false;
	}

	void dvidedata(double testratio) {
		testnum = M * testratio;
		test.clear();
		test.resize(testnum);
		int node1, node2;
		e.seed(time(0));
		uniform_int_distribution<int> u1(0, nsize - 1);
		for (int i = 0; i < testnum;) {
			node1 = u1(e);
			if (n[node1].size() > 1) {
				uniform_int_distribution<int> u2(0, n[node1].size() - 1);
				int pos = u2(e);
				node2 = n[node1][pos];
				if (connected(node1, node2)) {
					test[i].push_back(node1);
					test[i].push_back(node2);
					n[node1].erase(n[node1].begin() + pos);
					for (int j = 0; j < n[node2].size(); j++) {
						if (n[node2][j] == node1) {
							n[node2].erase(n[node2].begin() + j);
						}
					}
					i++;
				}
			}
		}
		cout << "data divided" << endl;
	}

	int CN(int n1, int n2) {
		int cn = 0;
		for (int i = 0; i < n[n1].size(); i++) {
			for (int j = 0; j < n[n2].size(); j++) {
				if (n[n1][i] == n[n2][j]) {
					cn++;
					break;
				}
			}
		}
		return cn;
	}

	double PE(int n1, int n2) {
		double pe = 0;
		vector<vector<int> > path2;
		vector<vector<int> > path3;
		int p2i = 0, p3i = 0;
		for (int i = 0; i < n[n1].size(); i++) {
			for (int j = 0; j < n[n2].size(); j++) {
				if (n[n1][i] == n[n2][j]) {
					path2.resize(p2i + 1);
					path2[p2i].push_back(n[n1][i]);
					p2i++;
				}
			}
		}
		for (int i = 0; i < n[n1].size(); i++) {
			for (int j = 0; j < n[n2].size(); j++) {
				for (int k = 0; k < n[n[n1][i]].size(); k++) {
					if (n[n[n1][i]][k] == n[n2][j]) {
						path3.resize(p3i + 1);
						path3[p3i].push_back(n[n1][i]);
						path3[p3i].push_back(n[n2][j]);
						p3i++;
						break;
					}
				}
			}
		}
		for (int i = 0; i < path2.size(); i++) {
			pe += IL1(n1, path2[i][0]) + IL1(path2[i][0], n2);
		}
		for (int i = 0; i < path3.size(); i++) {
			pe += 0.5*(IL1(n1, path3[i][0]) + IL1(path3[i][0], path3[i][1]) + IL1(path3[i][1], n2));
		}
		return pe - IL1(n1, n2);

	}

	double AA(int n1, int n2) {
		double aa = 0;
		for (int i = 0; i < n[n1].size(); i++) {
			for (int j = 0; j < n[n2].size(); j++) {
				if (n[n1][i] == n[n2][j]) {
					aa += 1 / log(n[n[n1][i]].size());
					break;
				}
			}
		}
		return aa;
	}

	double RA(int n1,int n2) {
		double ra = 0;
		for (int i = 0; i < n[n1].size(); i++) {
			for (int j = 0; j < n[n2].size(); j++) {
				if (n[n1][i] == n[n2][j]) {
					ra += 1 / n[n[n1][i]].size();
					break;
				}
			}
		}
		return ra;
	}

	double LNBC(int n1, int n2) {

	}

	void clear(queue<int>& q) {
		queue<int> empty;
		swap(empty, q);
	}

	double pn(int node, int maxlevel) {
		double addk = 0;
		int neighbor;//邻居
		clear(q);
		q.push(node);
		visited.clear();
		visited.resize(nsize, false);
		visited[node] = true;
		level.clear();
		level.resize(nsize, 0);
		while (!q.empty())
		{
			int v = q.front(); //取出队头的节点
			addk += n[v].size();
			q.pop();
			for (int i = 0; i < n[v].size(); i++) {
				neighbor = n[v][i];
				if (!visited[neighbor]) {
					level[neighbor] = level[v] + 1;
					if (level[neighbor] <= maxlevel) {
						q.push(neighbor);
						visited[neighbor] = true;
					}
				}
			}
		}
		return (double)n[node].size() / addk;
	}

	void countAUC(int num,int method) {//method:1.cn;2.pe
		int n11, n12;//测试边节点
		int n21, n22;//不存在边节点
		e.seed(time(0));
		uniform_int_distribution<int> v1(0, test.size() - 1);
		uniform_int_distribution<int> v2(0, nsize - 1);
		double s1, s2;
		double t = 0;//测试边相似度比不存在边大
		for (int i = 0; i < num; i++) {
			//随机选择测试边和不存在边各一条，分别计算相似度比较
			int testn = v1(e);
			n11 = test[testn][0];
			n12 = test[testn][1];
			n21 = v2(e);
			do
			{
				n22 = v2(e);
			} while (isNeighbor(n21, n22));
			switch (method)
			{
			case 1:
				s1 = PE(n11, n12);//测试边
				s2 = PE(n21, n22);//不存在边
				cout << "PE" << endl;
			case 2:
				s1 = CN(n11, n12);
				s2 = CN(n21, n22);
				cout << "CN" << endl;
			case 3:
				s1 = AA(n11, n12);
				s2 = AA(n21, n22);
				cout << "AA" << endl;
			case 4:
				s1 = RA(n11, n12);
				s2 = RA(n21, n22);
				cout << "RA" << endl;
			}
			if (s1 > s2) {
				t++;
			}
			else if (s1 == s2) {
				t += 0.5;
			}
		}
		AUC = t / num;
		cout << "AUC:"<<AUC << endl;
	}

	bool isNeighbor(int n1, int n2) {
		for (int i = 0; i < tempn[n1].size(); i++) {
			if (tempn[n1][i] == n2) {
				return true;
			}
		}
		return false;
	}

	double IL1(int na, int nb) {
		int ka = n[na].size();
		int kb = n[nb].size();
		double c = 1;
		for (int i = 0; i <= kb - 1; i++) {
			c *= (double)(M - ka - i) / (M - i);
		}
		return -log2(1 - c);
	}



};

int main(int argc, char **argv) {
	Graph g;
	//g.readdata("F:/data/polblogs.txt");
	//g.readdata("F:/data/football.txt");
	//g.readdata("F:/data/hep-th.txt");
	//g.readdata("F:/data/hep-th.txt");
	//g.readdata("F:/data/test.txt");
	g.readdata("F:/data/lp_data/Celegans.txt");
	g.dvidedata(0.1);
	//g.countAUC(100,1);
	g.countAUC(100,2);
	system("pause");
}