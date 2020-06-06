library(tidyverse)
library(tidytext)
library(shiny)
library(forcats)
library(scales)
library(arules)
library(arulesViz)
library(reticulate)

# importing data
chat_df <- read_csv("dota-2-matches\\chat.csv")
players_df <- read_csv("dota-2-matches\\players.csv")
heroes_df <- read_csv("dota-2-matches\\hero_names.csv")
match_df <-  read_csv("dota-2-matches\\match.csv")

heroes_df <- heroes_df %>% arrange(desc(localized_name))
# for use in full DF - sentiment
some_fun <- function(x) {
    if (x < 5) {
        return(x)
    }
    return(x - 123)
}
players_df$player_slot <-sapply(players_df$player_slot, some_fun)


# making full dataframe
full_df <- chat_df %>% inner_join(players_df, by=c("match_id"="match_id", "slot"="player_slot")) %>%
    select(match_id, key, slot, hero_id) %>%
    inner_join(heroes_df, by=c("hero_id"="hero_id")) %>%
    inner_join(match_df, by=c("match_id" = "match_id")) %>% select(match_id, key, hero_id, localized_name, slot, radiant_win) %>%
    mutate(side = ifelse(slot <= 5, "Radiant", "Dire")) %>%
    mutate(winner = case_when(
        (side == "Radiant" & radiant_win == TRUE) ~ TRUE,
        (side == "Radiant" & radiant_win == FALSE) ~ FALSE,
        (side == "Dire" & radiant_win == TRUE) ~ FALSE,
        TRUE ~ TRUE)
    )

# df of winning matches
winner_df <- full_df %>% filter(winner==TRUE)

winner_df_words <- winner_df %>%
    unnest_tokens(word, key) %>%
    anti_join(stop_words)

winner_df_counts <- winner_df_words %>% count(word) %>% mutate(prop = n/sum(n)) %>%  arrange(desc(n)) %>% mutate(result = "win")

# df of losing matches
loser_df <- full_df %>% filter(winner==FALSE)

loser_df_words <- loser_df %>%
    unnest_tokens(word, key) %>%
    anti_join(stop_words)

loser_df_counts <- loser_df_words %>% count(word) %>% mutate(prop = n/sum(n)) %>% arrange(desc(n)) %>% mutate(result = "loss")

# concatenated dataframe
top_10_each <- rbind(winner_df_counts[1:10,], loser_df_counts[1:10,])

# dota rules apriori
team_df <- read_csv("Hero_Drafting.csv")
items <- strsplit(as.character(team_df$localized_name), ", ")
transactions_dota = as(items, Class = 'transactions')
dota_rules <- apriori(transactions_dota, parameter = list(support = 0.001, confidence = 0.025, minlen = 2))
dota_rules <- dota_rules[!is.redundant(dota_rules)]

# win rates
hero_win_rates <- full_df %>% group_by(localized_name) %>% summarise(win_rate = sum(winner)/n())
hero_pick_rates <- full_df %>% group_by(localized_name) %>% summarise(pick_number = n()) %>% mutate(pick_rate = pick_number/sum(pick_number))


# sourcing python for neural network
source_python('hero_prediction_neural_net.py')

ui <- navbarPage(title="DOTA 2 Analysis",
                 tabPanel(title = "Multi-Hero Analysis", icon = icon("users", class = NULL, lib = "font-awesome"),
                          sidebarLayout(
                              sidebarPanel(
                                  selectInput("heroes", "Hero Selection", unique(heroes_df$localized_name), multiple = TRUE, selected = c("Ember Spirit", "Witch Doctor")),
                              ), # sidebarpanel analysis
                              mainPanel(
                                  tabsetPanel(
                                      tabPanel(title ="Hero Stats", icon = icon("user", class = NULL, lib = "font-awesome"),
                                               plotOutput("win_rate"),
                                               plotOutput("pick_rate")
                                      ), # analysis tag in mainpanel analysis
                                      tabPanel(title = "Hero Association Rules",
                                               tabsetPanel(
                                                   tabPanel(title ="Rules by Support (common picks)", icon = icon("users", class = NULL, lib = "font-awesome"),
                                                            textOutput("support_expl"),
                                                            plotOutput("assoc_rules_support")),
                                                   tabPanel(title ="Rules by Confidence (strong implied picks)", icon = icon("users", class = NULL, lib = "font-awesome"),
                                                            textOutput("confidence_expl"),
                                                            plotOutput("assoc_rules_confidence")),
                                                   tabPanel(title ="Rules by Lift (niche picks)", icon = icon("users", class = NULL, lib = "font-awesome"),
                                                            textOutput("lift_expl"),
                                                            plotOutput("assoc_rules_lift"))
                                               )
                                      ) # analysis tag in mainpanel analysis
                                  ) # analysis mainpanel navbarmenu
                              ) # mainpanel analysis
                          ), #analysis sidebarlayout
                 ), # Analysis tab
                 tabPanel(title = "Match Analysis", icon = icon("steam-symbol", class = NULL, lib = "font-awesome"),
                          sidebarLayout(
                              sidebarPanel(
                                  radioButtons("win_loss", "Win/Loss", choices = c("Win", "Loss", "All"), selected="All")
                              ), # analysis tab sidebar panel
                              mainPanel(
                                  plotOutput("win_loss_words_plot")
                              ) # analysis tab main panel
                          ) # anaylsis tab sidebar layout
                 ), # Sentiment Analysis
                 tabPanel(title = "Neural Network", icon = icon("project-diagram", class = NULL, lib = "font-awesome"),
                          sidebarLayout(
                              sidebarPanel(
                                  selectInput("classifier", "Choose a Neural Network Classifier", choices = c("LSTM", "FFNN"), selected = "LSTM"),
                                  selectizeInput("heroes_neural_net", "Select 9 heroes", choices = sort(heroes_df$localized_name), selected = sort(heroes_df$localized_name)[1:9], options = list(maxItems=9), multiple=TRUE)
                              ), # sidepanel neural net
                              mainPanel(
                                  plotOutput("neural_net_plot"),
                                  tableOutput("neural_net_table")
                              )
                          ) # sidebarlayout neural net
                 ) # Neural Net tab
)

server <- function(input, output, session) {
    hero_classifier <- reactive({input$classifier})
    heroes_var_total <- reactive({input$heroes_neural_net})
    heroes_var_1 <- reactive({input$heroes_neural_net[1]})
    heroes_var_2 <- reactive({input$heroes_neural_net}[2])
    heroes_var_3 <- reactive({input$heroes_neural_net}[3])
    heroes_var_4 <- reactive({input$heroes_neural_net}[4])
    heroes_var_5 <- reactive({input$heroes_neural_net}[5])
    heroes_var_6 <- reactive({input$heroes_neural_net}[6])
    heroes_var_7 <- reactive({input$heroes_neural_net}[7])
    heroes_var_8 <- reactive({input$heroes_neural_net}[8])
    heroes_var_9 <- reactive({input$heroes_neural_net}[9])
    r_results <- reactive({run_neural_network(hero_classifier(), heroes_var_1(),heroes_var_2(),heroes_var_3(),heroes_var_4(),heroes_var_5(),heroes_var_6(),heroes_var_7(),heroes_var_8(),heroes_var_9())})
    output$win_loss_words_plot <- renderPlot({
        if (input$win_loss == "All") {
            ggplot(top_10_each, aes(reorder(word, -prop), prop, fill=result)) +
                geom_bar(stat="identity", position="dodge") +
                ggtitle("Common Words in Winning Games") +
                xlab("Words") +
                ylab("Word Prevelance (as percentage)")
        }
        else if (input$win_loss == "Win") {
            ggplot(winner_df_counts[1:10,], aes(reorder(word, -prop), prop)) +
                geom_bar(stat="identity", fill='#00BFC4') +
                ggtitle("Common Words in Winning Games") +
                xlab("Words") +
                ylab("Word Prevelance (as percentage)")
        }
        else {
            ggplot(loser_df_counts[1:10,], aes(reorder(word, -prop), prop)) +
                geom_bar(stat="identity", fill='#F8766D') +
                ggtitle("Common Words in Winning Games") +
                xlab("Words") +
                ylab("Word Prevelance (as percentage)")
        }
    })
    
    output$win_rate <- renderPlot({
        ggplot(hero_win_rates, aes(x=win_rate)) +
            # geom_histogram(aes(y=..density..)) +      # Histogram with density instead of count on y-axis +
            geom_density(alpha=0.2, aes(fill="#4271AE")) + # Overlay with transparent density plot
            geom_vline(xintercept = hero_win_rates %>% filter(localized_name %in% input$heroes) %>% select(win_rate) %>%  pull(),
                       col=hue_pal()(length(input$heroes))) +
            annotate(geom="text", x = hero_win_rates %>% filter(localized_name %in% input$heroes) %>% select(win_rate) %>%  pull() + 0.0015,
                     y=rep(3, length(input$heroes)),
                     label = hero_win_rates %>% filter(localized_name %in% input$heroes) %>% select(localized_name) %>%  pull(),
                     angle=90,
                     size = 3,
                     col=hue_pal()(length(input$heroes))) +
            theme(legend.position = "none") +
            ggtitle("Win Rate")
    }, height = 400, width = 800)
    
    output$pick_rate <- renderPlot({
        ggplot(hero_pick_rates, aes(x=pick_rate)) +
            # geom_histogram(aes(y=..density..)) +      # Histogram with density instead of count on y-axis +
            geom_density(alpha=0.2, aes(fill="#00BFC4")) + # Overlay with transparent density plot
            geom_vline(xintercept = hero_pick_rates %>% filter(localized_name %in% input$heroes) %>% select(pick_rate) %>%  pull(),
                       col=hue_pal()(length(input$heroes))) +
            annotate(geom="text", x = hero_pick_rates %>% filter(localized_name %in% input$heroes) %>% select(pick_rate) %>%  pull() + 0.0003,
                     y=rep(20, length(input$heroes)),
                     label = hero_pick_rates %>% filter(localized_name %in% input$heroes) %>% select(localized_name) %>%  pull(),
                     angle=90,
                     size = 3,
                     col=hue_pal()(length(input$heroes))) +
            theme(legend.position = "none") +
            ggtitle("Pick Rate")
    }, height = 400, width = 800)
    
    output$assoc_rules_support <- renderPlot({
        hero_rules = subset(dota_rules, items %in% input$heroes)
        hero_rules_by_support = sort(hero_rules, by="support")
        plot(hero_rules_by_support[1:20], method="graph", size=c(15,15))
    }, height = 800, width = 1000)
    
    output$support_expl <- renderText({
        "Support(A, B) = P(A & B):
    The probability of the n heroes occurring together in the same game\n
    Support is the popularity of item combinations, a very simple measure that is is strongly effected by individual hero popularity, and does not tell us much about association rules.\n
    Consider a situation where a certain hero was picked every game, regardless of any synergy between heroes - this would have a support of 1 with every hero, which is not very informative"
    })
    output$assoc_rules_confidence <- renderPlot({
        hero_rules = subset(dota_rules, items %in% input$heroes)
        hero_rules_by_confidence = sort(hero_rules, by="confidence")
        plot(hero_rules_by_confidence[1:20], method="graph")
    }, height = 800, width = 1000)
    
    output$confidence_expl <- renderText({
        "Confidence(A, B) = P(A & B)/P(A):
    The probability that hero B will be picked, given that hero A has been picked.\n
    Unlike support, this is a measure of the strength of the interactions, though it is also affected by indivual hero popularity.\n
    Consider a situation where hero Y is picked nearly every game, regardless of hero combinations, and hero Z is not picked often, but is almost always picked after hero X, the confidence of the association rule between hero X --> hero Y, and hero X --> hero Z, could be very similar"
    })
    output$assoc_rules_lift <- renderPlot({
        hero_rules = subset(dota_rules, items %in% input$heroes)
        hero_rules_by_lift = sort(hero_rules, by="lift")
        plot(hero_rules_by_lift[1:20], method="graph")
    }, height = 800, width = 1000)
    
    output$lift_expl <- renderText({
        "Lift(A, B) = P(A & B)/(P(A)*P(B)):
    The increased value of the interation between the heroes \n
    This is a good measure of synergy, since it scales for individual hero popularity. \n
    Consider a situation where hero Y is very rarely picked, but is almost always picked if hero X has already been picked. The lift would be very high for interaction hero X --> hero Y, \n
    Consider another situation, where hero Z is picked an average amount, but is almost always picked if hero X has already been picked, the lift for the rule hero X --> hero Z would be fairly high, but not as high as the rule hero X --> hero Y"
    })
    
    output$neural_net_plot <- renderPlot({
        if (length(heroes_var_total()) == 9) {
            hero_names <- as.vector(r_results()[[1]])
            hero_probs <- as.vector(r_results()[[2]])
            neural_net_results <- data.frame(hero_names, hero_probs) %>% arrange(desc(hero_probs))
            ggplot(neural_net_results, aes(fct_reorder(neural_net_results$hero_names, neural_net_results$hero_probs, .desc = TRUE), hero_probs)) + geom_bar(stat='identity', aes(fill=hero_names)) + 
                theme(axis.title.x=element_blank(),
                      axis.title.y=element_blank())
        }
    })
    
    output$neural_net_table <- renderTable({
        if (length(heroes_var_total()) == 9) {
            hero_names <- as.vector(r_results()[[1]])
            hero_probs <- as.vector(r_results()[[2]])
            neural_net_results <- data.frame(hero_names, hero_probs) %>% arrange(desc(hero_probs))
            neural_net_results
        }
    }, digits = 5)
}

shinyApp(ui, server)


