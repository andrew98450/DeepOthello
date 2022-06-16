/*
	¶Â¥Õ´Ñ
	§@ªÌ:Huang Andrew 
*/
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <torch/torch.h>
using namespace std; 
struct Point
{
	int x;
	int y;
};
class DRL : public torch::nn::Module
{
protected:
	torch::nn::Conv2d conv = nullptr;
	torch::nn::Conv2d conv2 = nullptr;
	torch::nn::Conv2d conv3 = nullptr;
	torch::nn::Conv2d conv4 = nullptr;
	torch::nn::Conv2d conv5 = nullptr;
	torch::nn::Conv2d conv6 = nullptr;

	torch::nn::Linear fc = nullptr;
	torch::nn::Linear fc2 = nullptr;
public:
	DRL(int action)
	{
		conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 2)));
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 2)));
		conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(20, 40, 2)));
		conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(40, 80, 2)));
		conv5 = register_module("conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(80, 100, 2)));
		conv6 = register_module("conv6", torch::nn::Conv2d(torch::nn::Conv2dOptions(100, 200, 2)));

		fc = register_module("fc", torch::nn::Linear(torch::nn::LinearOptions(200, 512)));
		fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(512, action)));
	}
	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(conv->forward(x));
		x = torch::relu(conv2->forward(x));
		x = torch::relu(conv3->forward(x));
		x = torch::relu(conv4->forward(x));
		x = torch::relu(conv5->forward(x));
		x = torch::relu(conv6->forward(x));
		x = torch::avg_pool2d(x, 2);

		x = x.view({ -1,200 });

		x = torch::relu(fc->forward(x));
		x = fc2->forward(x);
		return x;
	}
};
class DQN  
{
protected:
	shared_ptr<DRL> drl;
	shared_ptr<DRL> drl_target;
public:
	DQN(int move_size)
	{
		drl = make_shared<DRL>(move_size);
		drl_target = make_shared<DRL>(move_size);
	}
	int choose_action(torch::Tensor x)
	{
		torch::optim::Adam adam(drl->parameters(), torch::optim::AdamOptions(0.01));
		for (int i = 0; i < 200; i++)
		{
			drl_target = drl;
			torch::Tensor output = drl->forward(x);
			torch::Tensor target = drl_target->forward(x).detach();
			torch::Tensor loss = torch::mse_loss(output, target);
			adam.zero_grad();
			loss.backward();
			adam.step();
		}
		torch::Tensor out = drl->forward(x);
		int index = out.argmax(1).item<int>();
		return index;
	}
};
void Init(int board[8][8])
{
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			board[i][j] = 0;
		}
	}
	board[3][3] = 2;
	board[3][4] = 1;
	board[4][3] = 1;
	board[4][4] = 2;
}
int Count(int board[8][8], int player)
{
	int count = 0;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if (board[i][j] == player)
			{
				count++;
			}
		}
	}
	return count;
}
void PrintGraph(int board[8][8])
{
	system("cls");
	printf("  1 2 3 4 5 6 7 8 [%s=%d %s=%d]\n", "O:Player", Count(board, 1), "X:AI", Count(board, 2));
	for (int i = 0; i < 8; i++)
	{
		printf("%d ", i + 1);
		for (int j = 0; j < 8; j++)
		{
			if (board[i][j] == 1)
			{
				printf("%c ", 'O');
			}
			else if (board[i][j] == 2)
			{
				printf("%c ", 'X');
			}
			else if (board[i][j] == 3)
			{
				printf("%c ", '+');
			}
			else
			{
				printf("%c ", '-');
			}
		}
		printf("\n");
	}
}
vector<Point> IsMove(int board[8][8], int pos[8][2], int player, int x, int y)
{
	int other_player = 0;
	Point rpoint;
	vector<Point> point;
	if (board[x][y] != 0)
	{
		return point;
	}

	board[x][y] = player;

	if (player == 1)
	{
		other_player = 2;
	}
	else
	{
		other_player = 1;
	}

	for (int i = 0; i < 8; i++)
	{
		int x_index = x, y_index = y;
		x_index += pos[i][0];
		y_index += pos[i][1];
		if (board[x_index][y_index] == other_player)
		{
			x_index += pos[i][0];
			y_index += pos[i][1];

			if (!(x_index < 8 && y_index < 8 && x_index >= 0 && y_index >= 0))
				continue;

			while (board[x_index][y_index] == other_player)
			{
				x_index += pos[i][0];
				y_index += pos[i][1];
				if (!(x_index < 8 && y_index < 8 && x_index >= 0 && y_index >= 0))
					break;
			}

			if (!(x_index < 8 && y_index < 8 && x_index >= 0 && y_index >= 0))
				continue;

			if (board[x_index][y_index] == player)
			{
				while (true)
				{
					x_index -= pos[i][0];
					y_index -= pos[i][1];
					if (x_index == x && y_index == y)
						break;
					rpoint.x = x_index;
					rpoint.y = y_index;
					point.push_back(rpoint);
				}
			}
		}
	}

	board[x][y] = 0;

	return point;
}
vector<Point> GetMove(int board[8][8], int pos[8][2], int player)
{
	Point p;
	vector<Point> point;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if (IsMove(board, pos, player, i, j).size() != 0)
			{
				p.x = i + 1;
				p.y = j + 1;
				point.push_back(p);
			}
		}
	}
	return point;
}
void ChangeBoard(int board[8][8], vector<Point> rp, int player)
{
	for (int i = 0; i < rp.size(); i++)
	{
		board[rp[i].x][rp[i].y] = player;
	}
}
void PromptIndex(int board[8][8], vector<Point> move)
{
	for (int k = 0; k < move.size(); k++)
	{
		board[move[k].x - 1][move[k].y - 1] = 3;
	}
}
void ClearPrompt(int board[8][8], vector<Point> move)
{
	for (int k = 0; k < move.size(); k++)
	{
		board[move[k].x - 1][move[k].y - 1] = 0;
	}
}
torch::Tensor BoardToTensor(int board[8][8])
{
	torch::Tensor t = torch::zeros({ 1,1,8,8 }, torch::kFloat);
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			t[0][0][i][j] = board[i][j];
		}
	}
	return t;
}
int Execute(int board[8][8], vector<Point> move)
{
	int index = 0;
	torch::Tensor input = BoardToTensor(board);
	DQN dqn(move.size());
	index = dqn.choose_action(input);
	return index;
}
int PlayerInput(int board[8][8], int pos[8][2], int i, int j)
{
	vector<Point> rp = IsMove(board, pos, 1, i - 1, j - 1);
	vector<Point> move = GetMove(board, pos, 1);
	if (move.size() == 0)
	{
		PrintGraph(board);
		return 0;
	}
	while (rp.size() == 0)
	{
		PromptIndex(board, move);
		PrintGraph(board);
		ClearPrompt(board, move);
		fflush(stdin);
		printf("Enter Point (xÁa¶b,y¾î¶b):");
		int val = scanf("%d,%d", &i, &j);
		rp = IsMove(board, pos, 1, i - 1, j - 1);
	}
	board[i - 1][j - 1] = 1;
	ChangeBoard(board, rp, 1);
	PrintGraph(board);
	return 1;
}
int AIInput(int board[8][8], int pos[8][2])
{
	vector<Point> move = GetMove(board, pos, 2);
	if (move.size() == 0)
	{
		PrintGraph(board);
		return 0;
	}
	int val = Execute(board, move);
	vector<Point> rp = IsMove(board, pos, 2, move[val].x - 1, move[val].y - 1);
	board[move[val].x - 1][move[val].y - 1] = 2;
	ChangeBoard(board, rp, 2);
	PrintGraph(board);
	return 1;
}
int IsFull(int board[8][8])
{
	int len = 64, count = 0;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if (board[i][j] != 0)
			{
				count++;
			}
		}
	}
	if (count == len)
		return 1;
	else
		return 0;
}
void StartGame(bool prompt)
{
	int i = 0, j = 0;
	int board[8][8];
	int pos[8][2] = { {0, 1}, {1, 1}, {1, 0}, {1, -1},
	{0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };
	Init(board);
	PrintGraph(board);
	while (true)
	{
		vector<Point> move = GetMove(board, pos, 1);
		
		if (prompt == true)
		{
			PromptIndex(board, move);
			PrintGraph(board);
			ClearPrompt(board, move);
			printf("Enter Point (xÁa¶b,y¾î¶b):");
		}
		else
		{
			PrintGraph(board);
			printf("Enter Point (xÁa¶b,y¾î¶b):");
		}

		int player = 0;
		if (move.size() != 0)
		{
			int val = scanf("%d,%d", &i, &j);
			player = PlayerInput(board, pos, i, j);
		}

		vector<Point> ai_move = GetMove(board, pos, 2);
		int ai = 0;
		if (ai_move.size() != 0)
		{
			ai = AIInput(board, pos);
		}

		if ((player == 0 && ai == 0) || IsFull(board) == 1)
		{
			if (Count(board, 1) > Count(board, 2))
			{
				printf("Player is Win...\n");
				system("pause");
			}
			else if (Count(board, 1) < Count(board, 2))
			{
				printf("AI is Win...\n");
				system("pause");
			}
			else
			{
				printf("Player And AI is Peace...\n");
				system("pause");
			}
			break;
		}
	}
}
int main(void)
{
	int select = 0;
	while (true)
	{
		fflush(stdin);
		printf("==================\n");
		printf("      Othello     \n");
		printf("==================\n");
		printf("1.Prompt Mode\n");
		printf("2.General Mode\n");
		printf(">>");
		int val = scanf("%d", &select);
		switch (select)
		{
		case 1:
			StartGame(true);
			break;
		case 2:
			StartGame(false);
			break;
		default:
			break;
		}
		system("cls");
	}
	return 0;
}
