#include <vector>
#include <ctime>
#include <iostream>
#include <random>
#include <fstream>  
#include <string>  
#include <sstream>
#include <cmath>
#include <queue>
#include <algorithm>
#include <io.h>

using namespace std;

typedef struct {
	int source;
	int dest;
	double weight;
	double score;
	bool isTest;
}link;

bool destLess(link l1, link l2) {
	return l1.dest < l2.dest;
}

bool scoreGreater(link l1, link l2) {
	return l1.score > l2.score;
}

class Network {
private:
	int nsize;//节点数量
	vector<int> nlist;
	int M;//边数
	int testnum;//测试边数
	int begin;//开始节点
	vector<vector<link> > n;//邻接表
	vector<link> linklist;//边集
	vector<link> templist;
	vector<vector<link> > tempn;
	vector<vector<link> > train;//训练集
	vector<link> test;//测试集
	vector<link> nolink;//不存在边集
	vector<string> files;
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

	void readdata(string filename, bool isWeighted) {
		ifstream in(filename);
		string line;
		if (in) // 有该文件  
		{
			int n1, n2, temp;
			double w = 1;
			n.clear();
			linklist.clear();
			link l;
			nsize = 0;
			M = 0;
			begin = 0;
			while (getline(in, line)) // line中不包括每行的换行符  
			{
				M++;
				istringstream is(line);
				if (isWeighted) {
					is >> n1 >> n2 >> w;
				}
				else {
					is >> n1 >> n2;
				}
				l.source = n1;
				l.dest = n2;
				l.weight = w;
				linklist.push_back(l);
				if ((n1 >= nsize) || (n2 >= nsize)) {
					nsize = ((n1 > n2) ? n1 : n2) + 1;
					n.resize(nsize);
				}
				n[n1].push_back(l);
				temp = l.source;
				l.source = l.dest;
				l.dest = temp;
				n[n2].push_back(l);
			}
			if (n[0].size() == 0) {
				begin = 1;
			}
			for (int i = 0; i < n.size(); i++) {
				sort(n[i].begin(), n[i].end(), destLess);

			}
			e.seed(time(0));
		}
		else // 没有该文件  
		{
		cout << "no such file" << endl;
		return;
		}
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
		int v;
		int t = 0;
		for (int i = 0; i < train[n1].size(); i++) {
			if (train[n1][i].dest != n2) {
				q.push(train[n1][i].dest);
				visited[train[n1][i].dest] = true;
			}
		}
		while (!q.empty()) {
			t++;
			v = q.front();
			q.pop();
			for (int i = 0; i < train[v].size(); i++) {
				int now = train[v][i].dest;
				if (!visited[now]) {
					if (now == n2) {
						return true;
					}
					q.push(now);
					visited[now] = true;
				}
			}
			/*if (t > 5000) {//限制搜索次数，加快速度
				return false;
			}*/
		}
		return false;
	}


	bool isNeighbor(int n1, int n2) {
		for (int i = 0; i < n[n1].size(); i++) {
			if (n[n1][i].dest == n2) {
				return true;
			}
		}
		for (int j = 0; j < n[n2].size(); j++) {
			if (n[n2][j].dest == n1) {
				return true;
			}
		}
		return false;
	}

	void dividedata() {
		int d;
		int n1, n2;
		link l;
		l.isTest = true;
		for (int i = 0; i < testnum;) {
			uniform_int_distribution<int> u1(0, templist.size() - 1);
			int pos = u1(e);
			l.source = n1 = templist[pos].source;
			l.dest = n2 = templist[pos].dest;
			if ((train[n1].size() > 1) && ((train[n2].size() > 1)) && (connected(n1, n2))) {
				templist.erase(templist.begin() + pos);
				test[i] = l;
				for (int j = 0; j < train[n1].size(); j++) {
					if (train[n1][j].dest == n2) {
						train[n1].erase(train[n1].begin() + j);
					}
				}
				for (int k = 0; k < train[n2].size(); k++) {
					if (train[n2][k].dest == n1) {
						train[n2].erase(train[n2].begin() + k);
					}
				}
				i++;
			}
		}
	}

	void clear(queue<int>& q) {
		queue<int> empty;
		swap(empty, q);
	}

	void deleteLink(string infile, string trainfile, double ratio) {
		ofstream out1(trainfile);
		readdata(infile, false);
		init(ratio);
		for (int i = 0; i < templist.size(); i++) {
			out1 << templist[i].source << " " << templist[i].dest << endl;
		}
		out1.close();
	}

	void getPart(string infile, string outfile, int num) {
		readdata(infile, false);
		ofstream out(outfile);
		nlist.clear();
		clear(q);
		int n1, n2;
		uniform_int_distribution<int> u(0, nsize - 1);
		do {
			n1 = u(e);
		} while (n[n1].size() <= 0);
		q.push(n1);
		visited.clear();
		visited.resize(nsize, false);
		visited[n1] = true;
		while (num > 0)
		{
			int v = q.front(); //取出队头的节点
			q.pop();
			nlist.push_back(v);
			for (int i = 0; i < n[v].size(); i++) {
				int now = n[v][i].dest;
				if (!visited[now]) {
					q.push(now);
					visited[now] = true;
				}
			}
			num--;
		}
		sort(nlist.begin(), nlist.end());
		for (int i = 0; i < linklist.size(); i++) {
			n1 = linklist[i].source;
			n2 = linklist[i].dest;
			if (binary_search(nlist.begin(), nlist.end(), n1) && binary_search(nlist.begin(), nlist.end(), n2)) {
				out << n1 << " " << n2 << endl;
			}
		}
		out.close();
	}


	void getWCC(string infile, string outfile) {//获取连通片
		readdata(infile, false);
		ofstream out(outfile);
		nlist.clear();
		clear(q);
		int n1, n2;
		uniform_int_distribution<int> u(0, nsize - 1);
		do {
			n1 = u(e);
		} while (n[n1].size() <= 0);
		q.push(n1);
		visited.clear();
		visited.resize(nsize, false);
		visited[n1] = true;
		while (!q.empty())
		{
			int v = q.front(); //取出队头的节点
			q.pop();
			nlist.push_back(v);
			for (int i = 0; i < n[v].size(); i++) {
				int now = n[v][i].dest;
				if (!visited[now]) {
					q.push(now);
					visited[now] = true;
				}
			}
		}
		sort(nlist.begin(), nlist.end());
		for (int i = 0; i < linklist.size(); i++) {
			n1 = linklist[i].source;
			n2 = linklist[i].dest;
			if (binary_search(nlist.begin(), nlist.end(), n1) && binary_search(nlist.begin(), nlist.end(), n2)) {
				out << lower_bound(nlist.begin(), nlist.end(), n1) - nlist.begin() << " " << lower_bound(nlist.begin(), nlist.end(), n2) - nlist.begin() << endl;
			}
		}
		out.close();
	}


	void sample(int num, string outfile) {//采样
		ofstream out(outfile);
		nlist.clear();
		clear(q);
		int n1, n2;
		uniform_int_distribution<int> u(0, nsize - 1);
		do {
			n1 = u(e);
		} while (n[n1].size() <= 0);
		q.push(n1);
		visited.clear();
		visited.resize(nsize, false);
		visited[n1] = true;
		while (num > 0)
		{
			int v = q.front(); //取出队头的节点
			q.pop();
			nlist.push_back(v);
			for (int i = 0; i < n[v].size(); i++) {
				int now = n[v][i].dest;
				if (!visited[now]) {
					q.push(now);
					visited[now] = true;
				}
			}
			num--;
		}
		sort(nlist.begin(), nlist.end());
		for (int i = 0; i < linklist.size(); i++) {
			n1 = linklist[i].source;
			n2 = linklist[i].dest;
			if (binary_search(nlist.begin(), nlist.end(), n1) && binary_search(nlist.begin(), nlist.end(), n2)) {
				out << lower_bound(nlist.begin(), nlist.end(), n1) - nlist.begin() << " " << lower_bound(nlist.begin(), nlist.end(), n2) - nlist.begin() << endl;
			}
		}
		out.close();
	}

	void makeUndirected(string infile, string outfile, bool isWeighted) {//转换为无向图
		ifstream in(infile);
		string line;
		if (in) // 有该文件  
		{
			int n1, n2, temp;
			double w = 1;
			n.clear();
			linklist.clear();
			nsize = 0;
			M = 0;
			begin = 0;
			link l;
			while (getline(in, line)) // line中不包括每行的换行符  
			{
				M++;
				istringstream is(line);
				if (isWeighted) {
					is >> n1 >> n2 >> w;
				}
				else {
					is >> n1 >> n2;
				}
				if ((n1 >= nsize) || (n2 >= nsize)) {
					nsize = ((n1 > n2) ? n1 : n2) + 1;
					n.resize(nsize);
				}
				if (n1 != n2) {
					if (isNeighbor(n1, n2)) {//边已存在
						for (int i = 0; i < n[n1].size(); i++) {
							if (n[n1][i].dest == n2) {
								n[n1][i].weight += w;
							}
						}
						for (int j = 0; j < n[n2].size(); j++) {
							if (n[n2][j].dest == n1) {
								n[n2][j].weight += w;
							}
						}
					}
					else {//边不存在
						l.source = n1;
						l.dest = n2;
						l.weight = w;
						n[n1].push_back(l);
						temp = l.source;
						l.source = l.dest;
						l.dest = temp;
						n[n2].push_back(l);
					}
				}
			}
			for (int i = 0; i < n.size(); i++) {
				sort(n[i].begin(), n[i].end(), destLess);
			}
			in.close();
			ofstream out(outfile);
			for (int i = 0; i < n.size(); i++) {
				for (int j = 0; j < n[i].size(); j++) {
					if (n[i][j].dest > i) {
						out << i << " " << n[i][j].dest << endl;
					}
				}
			}
			out.close();
		}
		else // 没有该文件  
		{
			cout << "no such file" << endl;
			return;
		}
		cout << outfile << " loaded" << endl;
	}

	void makeAllUndirected(const char* path) {
		char path2[30];
		strcpy_s(path2, path);
		strcat_s(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		for (int f = 0; f < files.size(); f++) {
			makeUndirected(fs + files[f], fs + files[f] + "_U", false);
		}
	}

	void makeAllWCC(const char* path) {
		char path2[30];
		strcpy_s(path2, path);
		strcat_s(path2, "*.txt");
		getAllFile(path2);
		string fs = path;
		for (int f = 0; f < files.size(); f++) {
			getWCC(fs + files[f], "F:/data/temp/wcc/" + files[f]);
		}
	}

};

int main(int argc, char **argv) {
	Network g;
	//g.deleteLink("F:/data/lp_data/Power.txt", "F:/data/lp_data/Power_train.txt", 0.1);
	g.makeAllWCC("F:/data/temp/");
	//g.getWCC("F:/data/temp/NetScience_U.txt", "F:/data/temp/NetScience_U_wcc.txt");
	//g.readdata("F:/data/temp/NetScience_379.txt", false);
	//g.sample(379, "F:/data/temp/NetScience.txt");
}