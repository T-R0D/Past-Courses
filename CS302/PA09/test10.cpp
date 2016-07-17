#include <iostream>
#include <string>

using namespace std;

#include "HashTable.cpp"

class TestData {
  public:
    TestData();
    void setKey(const string& newKey);
    string getKey() const;
    int getValue() const;
    static unsigned int hash(const string& str);

  private:
    string key;
    int value;
    static int count;
};

int TestData::count = 0;

TestData::TestData() : value(++count) {
}

void TestData::setKey(const string& newKey) {
    key = newKey;
}

string TestData::getKey() const {
    return key;
}

int TestData::getValue() const {
    return value;
}

unsigned int TestData::hash(const string& str) {
    unsigned int val = 0;

    for (int i = 0; i < str.length(); ++i) {
	val += str[i];
    }

    return val;
}


void print_help() {
    cout << endl << "Commands:" << endl;
    cout << "  H   : Help (displays this message)" << endl;
    cout << "  +x  : Insert (or update) data item with key x" << endl;
    cout << "  -x  : Remove the data element with the key x" << endl;
    cout << "  ?x  : Retrieve the data element with the key x" << endl;
    cout << "  E   : Empty table?" << endl;
    cout << "  C   : Clear the table" << endl;
    cout << "  Q   : Quit the test program" << endl;
}

int main(int argc, char **argv) {
    HashTable<TestData, string> table(7);

    print_help();

    do {
	table.showStructure();

	cout << endl << "Command: ";
	char cmd;
	cin >> cmd;

	TestData item;
	if (cmd == '+' || cmd == '?' || cmd == '-') {
	    string key;
	    cin >> key;
	    item.setKey(key);
	}

	switch (cmd) {
	  case 'H':
	  case 'h':
	    print_help();
	    break;

	  case '+':
	    table.insert(item);
	    cout << "Inserted data item with key ("
		 << item.getKey() << ") and value ("
		 << item.getValue() << ")" << endl;
	    break;

	  case '-':
	    if (table.remove(item.getKey())) {
		cout << "Removed data item with key ("
		     << item.getKey() << ")" << endl;
	    } else {
		cout << "Could not remove data item with key ("
		     << item.getKey() << ")" << endl;
	    }
	    break;

	  case '?':
	    if (table.retrieve(item.getKey(), item)) {
		cout << "Retrieved data item with key ("
		     << item.getKey() << ") and value ("
		     << item.getValue() << ")" << endl;
	    } else {
		cout << "Could not retrieve data item with key ("
		     << item.getKey() << ")" << endl;
	    }
	    break;

	  case 'C':
	  case 'c':
	    cout << "Clear the hash table" << endl;
	    table.clear();
	    break;

	  case 'E':
	  case 'e':
	    cout << "Hash table is "
		 << (table.isEmpty() ? "" : "NOT")
		 << " empty" << endl;
	    break;

	  case 'Q':
	  case 'q':
	    return 0;

	  default:
	    cout << "Invalid command" << endl;
	}
    } while (1);

    return 0;
}

