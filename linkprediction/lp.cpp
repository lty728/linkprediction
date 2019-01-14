#include <vector>
#include <ctime>
#include <iostream>
#include <random>
#include <fstream>  
#include <string>  
#include <sstream>
#include <cmath>
#include <queue>
#include <io.h>

using namespace std;

class Graph {
private:
	int nsize;//节点数量
	int M;//边数
	int testnum;//测试边数
	int begin;
	vector<vector<int> > n; //二维向量存储每个节点的邻接节点
	vector<vector<int> > train;//训练集
	vector<vector<int> > test;//测试集
	vector<vector<int> > linklist;//边集
	vector<vector<int> > templist;
	vector<string> files;
	double AUC;
	queue<int> q; //搜索队列
	vector<bool> visited; //访问标志
	vector<int> level;//广度优先遍历层数
	default_random_engine e;//随机数引擎


public:

	void getAllFile(const char* path) {
		_finddata_t file;
		intptr_t lf;
		//输入文件夹路径
		if ((lf = _findfirst(path, &file)) == -1)  //若将*.*改为*.txt，则会输出该文件夹下所有的txt文件名
			cout << "Not Found!" << endl;
		else {
			//输出文件名
			files.push_back(file.name);
			while (_findnext(lf, &file) == 0) {
				files.push_back(file.name);
			}
		}
		_findclose(lf);
	}

	void readdata(string filename) {
		ifstream in(filename);
		string line;
		if (in) // 有该文件  
		{
			int n1, n2;
			n.clear();
			linklist.clear();
			nsize = 0;
			M = 0;
			begin = 0;
			vector<int> l(2);
			while (getline(in, line)) // line中不包括每行的换行符  
			{
				M++;
				istringstream is(line);
				is >> n1 >> n2;
				l[0] = n1;
				l[1] = n2;
				linklist.push_back(l);
				if ((n1 >= nsize) || (n2 >= nsize)) {
					nsize = ((n1 > n2) ? n1 : n2) + 1;
					n.resize(nsize);

				}
				n[n1].push_back(n2);
				n[n2].push_back(n1);
			}
			if (n[0].size() == 0) {
				begin = 1;
			}
			e.seed(time(0));
		}
		else // 没有该文件  
		{
			cout << "no such file" << endl;
			return;
		}
		cout << filename << " loaded" << endl;
	}

	void init(double ratio) {
		testnum = M * ratio;
		train.clear();
		train.assign(n.begin(), n.end());
		test.clear();
		test.resize(testnum);
		templist.clear();
		templist.assign(linklist.begin(), linklist.end());
		dividedata();
	}


	bool connected(int n1, int n2) {//用于验证删掉一条边后网络是否连通
		visited.clear();
		visited.resize(nsize, false);
		visited[n1] = true;
		queue<int> q;
		for (int i = 0; i < train[n1].size(); i++) {
			if (train[n1][i] != n2) {
				q.push(train[n1][i]);
				visited[train[n1][i]] = true;
			}
		}
		while (!q.empty()) {
			int v = q.front();
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				int now = train[v][i];
				if (!visited[now]) {
					if (now == n2) {
						return true;
					}
					q.push(now);
					visited[now] = true;
				}
			}
		}
		return false;
	}

	double get_cluster() {
		double sum = 0;
		int ni;
		double ki;
		for (int i = 0; i < n.size(); i++) {
			ni = 0;
			ki = n[i].size();
			if (n[i].size() > 1) {
				for (int j = 0; j < n[i].size() - 1; j++) {
					for (int k = j + 1; k < n[i].size(); k++) {
						if (isNeighbor(n[i][j], n[i][k])) {
							ni++;
						}
					}
				}
				sum += 2 * ni / (ki*(ki - 1));
			}
		}
		return sum / n.size();
	}

	void dividedata() {
		int n1, n2;
		for (int i = 0; i < testnum;) {
			uniform_int_distribution<int> u1(0, templist.size() - 1);
			int pos = u1(e);
			n1 = templist[pos][0];
			n2 = templist[pos][1];
			if ((train[n1].size() > 1)&&((train[n2].size() > 1))&&(connected(n1, n2))) {
				templist.erase(templist.begin() + pos);
				test[i].push_back(n1);
				test[i].push_back(n2);
				for (int j = 0; j < train[n1].size();j++) {
					if (train[n1][j] == n2) {
						train[n1].erase(train[n1].begin() + j);
					}
				}
				for (int k = 0; k < train[n2].size(); k++) {
					if (train[n2][k] == n1) {
						train[n2].erase(train[n2].begin() + k);
					}
				}
				i++;
			}
		}
	}

	int CN(int n1, int n2) {
		int cn = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i] == train[n2][j]) {
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
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i] == train[n2][j]) {
					path2.resize(p2i + 1);
					path2[p2i].push_back(train[n1][i]);
					p2i++;
				}
			}
		}
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				for (int k = 0; k < train[train[n1][i]].size(); k++) {
					if (train[train[n1][i]][k] == train[n2][j]) {
						path3.resize(p3i + 1);
						path3[p3i].push_back(train[n1][i]);
						path3[p3i].push_back(train[n2][j]);
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
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i] == train[n2][j]) {
					aa += 1 / log(train[train[n1][i]].size());
					break;
				}
			}
		}
		return aa;
	}

	double RA(int n1, int n2) {
		double ra = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			for (int j = 0; j < train[n2].size(); j++) {
				if (train[n1][i] == train[n2][j]) {
					ra += 1 / (double)train[train[n1][i]].size();
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
			addk += train[v].size();
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				neighbor = train[v][i];
				if (!visited[neighbor]) {
					level[neighbor] = level[v] + 1;
					if (level[neighbor] <= maxlevel) {
						q.push(neighbor);
						visited[neighbor] = true;
					}
				}
			}
		}
		return (double)train[node].size() / addk;
	}

	int distance(int n1, int n2) {
		int distance = 0;
		clear(q);
		q.push(n1);
		visited.clear();
		visited.resize(nsize, false);
		level.clear();
		level.resize(nsize, 0);
		visited[n1] = true;
		while (!q.empty())
		{
			int v = q.front(); //取出队头的节点
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				int now = train[v][i];
				if (!visited[now]) {
					level[now] = level[v] + 1;
					if (now == n2) {
						return level[now];
					}
					else {
						q.push(now);
						visited[now] = true;
					}
				}
			}
		}
	}
	
	double newlp(int n1, int n2, int l) {
		double p1 = pn(n1, l);
		double p2 = pn(n2, l);
		int d = distance(n1, n2);
		return (-p1*log(p1) - p2*log(p2)) / (d - 1);
	}



	double countAUC(int num, int method) {//method:1.cn;2.pe
		int n11, n12;//测试边节点
		int n21, n22;//不存在边节点
		uniform_int_distribution<int> v1(0, test.size() - 1);
		uniform_int_distribution<int> v2(begin, nsize - 1);
		double s1, s2;
		double t = 0;//测试边相似度比不存在边大
		for (int i = 0; i < num; i++) {
			//随机选择测试边和不存在边各一条，分别计算相似度比较
			int testn = v1(e);
			n11 = test[testn][0];
			n12 = test[testn][1];
			do
			{
				n21 = v2(e);
				n22 = v2(e);
			} while (isNeighbor(n21, n22));
			switch (method)
			{
			case 1:
				s1 = CN(n11, n12);//测试边
				s2 = CN(n21, n22);//不存在边
				break;
			case 2:
				s1 = AA(n11, n12);
				s2 = AA(n21, n22);
				break;
			case 3:
				s1 = RA(n11, n12);
				s2 = RA(n21, n22);
				break;
			case 4:
				s1 = PE(n11, n12);
				s2 = PE(n21, n22);
				break;
			case 5:
				s1 = newlp(n11, n12, 3);
				s2 = newlp(n21, n22, 3);
				break;
			}
			if (s1 > s2) {
				t++;
			}
			else if (s1 == s2) {
				t += 0.5;
			}
		}
		AUC = t / num;
		return AUC;
	}

	bool isNeighbor(int n1, int n2) {
		for (int i = 0; i < n[n1].size(); i++) {
			if (n[n1][i] == n2) {
				return true;
			}
		}
		return false;
	}

	double IL1(int na, int nb) {
		int ka = train[na].size();
		int kb = train[nb].size();
		double c = 1;
		for (int i = 0; i <= kb - 1; i++) {
			c *= (double)(M - ka - i) / (M - i);
		}
		return -log2(1 - c);
	}


	void tries(int times,const char* path) {
		char path2[30];
		strcpy(path2, path);
		strcat(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		double result = 0;
		clock_t start, end;
		double ai;
		for (int f = 0; f < files.size(); f++) {
			readdata(fs.append(files[f]));
			for (int m = 1; m <= 5; m++) {
				for (int t = 1; t <= times; t++) {
					result = 0;
					cout << t << ":";
					start = clock();
					init(0.1);
					ai = countAUC(2000, m);
					result += ai;
					end = clock();
					cout << ai << " ";
					cout << end - start << "ms" << endl;
				}
				result = result / times;
				cout << result << endl;
			}
		}			
	}

	void undirect(string filename)

};

int main(int argc, char **argv) {
	Graph g;
	g.tries(100, "F:\\data\\lp_data\\");//1.PE;2.CN;3.AA;4.RA;5.my
	//g.init(0.1);
	//cout << g.get_cluster();
	//system("pause");
}